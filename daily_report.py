import os
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import traceback
import asyncio
import aiohttp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch 
import gc

# 1. CẤU HÌNH HỆ THỐNG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

VLLM_API_BASE = os.getenv("LLM_API_BASE")
VLLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
SPREADSHEET_ID = os.getenv("GOOGLE_SPREADSHEET_ID")
google_creds_filename = os.getenv("GOOGLE_SHEETS_CRED_FILE")
GOOGLE_CREDS = os.path.join(BASE_DIR, google_creds_filename) if not os.path.isabs(google_creds_filename) else google_creds_filename

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
receiver_raw = os.getenv("RECEIVER_EMAIL", "")
RECEIVER_EMAILS = [e.strip() for e in receiver_raw.split(",") if e.strip()]
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"

# 2. CÁC HÀM XỬ LÝ DỮ LIỆU
def get_sheet_data(service, range_name):
    """Đọc dữ liệu từ Google Sheets"""
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
    values = result.get('values', [])
    if not values: return pd.DataFrame()
    return pd.DataFrame(values[1:], columns=values[0])

async def fetch_summary(session, questions, semaphore, retries=3):
    """Gọi AI tóm tắt nội dung (Async) - Có log lỗi và retry"""
    async with semaphore:
        combined_questions = " | ".join(questions)
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": VLLM_MODEL_NAME,
            "messages": [
                {
                    "role": "system", 
                    "content": "Bạn là chuyên viên tổng hợp thông tin. Dựa vào các câu hỏi của thí sinh, hãy tóm tắt đầy đủ thành 1 đoạn văn ngắn (khoảng 30 chữ) liệt kê rõ các ngành học hoặc vấn đề cụ thể họ quan tâm."
                },
                {"role": "user", "content": f"Danh sách câu hỏi: {combined_questions}"}
            ],
            "temperature": 0.1,
            "max_tokens": 150 
        }
        
        # Thêm cơ chế retry để xử lý các lỗi nghẽn mạng tạm thời
        for attempt in range(retries):
            try:
                async with session.post(f"{VLLM_API_BASE}/chat/completions", headers=headers, json=payload, timeout=60) as response:
                    # Ném lỗi nếu HTTP status code không phải 200 (ví dụ: 400, 500, 503)
                    response.raise_for_status() 
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                # Ghi nhận log thực tế vào console thay vì nuốt lỗi
                print(f"[Lỗi AI - Thử lại lần {attempt + 1}/{retries}] Chi tiết: {e}")
                if attempt == retries - 1:
                    # Trả về chuỗi có chứa lỗi để ghi vào Excel, giúp bạn dễ trace sau này
                    return f"Lỗi phân tích AI: {str(e)}"
                await asyncio.sleep(2) # Chờ 2 giây trước khi thử lại

async def process_all_summaries(questions_list):
    """Quản lý luồng gọi AI hàng loạt"""
    # GIẢM CONCURRENCY: mức an toàn cho vLLM (ví dụ: 5 hoặc 10)
    semaphore = asyncio.Semaphore(5) 
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_summary(session, q, semaphore) for q in questions_list]
        return await asyncio.gather(*tasks)

def update_cloud_summary_sheet(service, df, target_date):
    """Lưu dữ liệu nối tiếp vào Google Sheets theo từng ngày trong tháng"""
    sheet_title = f"tong_hop_thong_tin_thang_{target_date.strftime('%m_%Y')}"
    
    # Kiểm tra/Tạo sheet nếu chưa có
    spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    sheets = [s.get('properties', {}).get('title') for s in spreadsheet.get('sheets', [])]
    
    if sheet_title not in sheets:
        body = {'requests': [{'addSheet': {'properties': {'title': sheet_title}}}]}
        service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()

    # Chuẩn bị dữ liệu ghi: Tiêu đề ngày -> Tên cột -> Dữ liệu -> 2 dòng trống
    all_data = [
        [f"--- BÁO CÁO NGÀY {target_date.strftime('%d/%m/%Y')} ---"],
        df.columns.tolist()
    ] + df.values.tolist() + [[""], [""]]
    
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{sheet_title}!A1",
        valueInputOption="USER_ENTERED",
        body={"values": all_data}
    ).execute()

# 3. HÀM GỬI EMAIL
def send_email_report(html_body, is_error=False):
    """Gửi email báo cáo hoặc báo lỗi"""
    subject = f"Báo Cáo Tuyển Sinh BLU - {datetime.now().strftime('%d/%m/%Y')}"
    if is_error: subject = f"[LỖI] {subject}"
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECEIVER_EMAILS)
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e: print(f"Lỗi gửi mail: {e}")

# 4. TIẾN TRÌNH CHÍNH
def main():
    print("Bắt đầu tiến trình tổng hợp...")
    creds = service_account.Credentials.from_service_account_file(GOOGLE_CREDS, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    service = build('sheets', 'v4', credentials=creds)
    
    target_date = datetime.now().date() - timedelta(days=1)
    month_year = target_date.strftime('%m_%Y')
    
    try:
        df_user = get_sheet_data(service, f"User_{month_year}!A:E")
        df_chat = get_sheet_data(service, f"Chat_{month_year}!A:F")
    except Exception as e:
        # Xử lý khi chưa tạo sheet tháng mới
        msg = f"<h2>Báo cáo ngày {target_date.strftime('%d/%m/%Y')}</h2><p>Chưa có dữ liệu. (Sheet <b>User_{month_year}</b> và <b>Chat_{month_year}</b> chưa tồn tại trên Google Sheets).</p>"
        send_email_report(msg)
        return

    if df_user.empty or df_chat.empty:
        send_email_report(f"<h2>Báo cáo ngày {target_date.strftime('%d/%m/%Y')}</h2><p>Không có dữ liệu tương tác nào trong trang tính.</p>")
        return

    # Tiền xử lý
    df_user['Thời gian'] = pd.to_datetime(df_user['Thời gian'], format='mixed')
    df_chat['Thời gian'] = pd.to_datetime(df_chat['Thời gian'], format='mixed')
    
    # Lọc theo ngày
    if not TEST_MODE:
        df_user = df_user[df_user['Thời gian'].dt.date == target_date]
        df_chat = df_chat[df_chat['Thời gian'].dt.date == target_date]

    # FIX LỖI 1: Kiểm tra rỗng sau khi lọc ngày
    if df_user.empty or df_chat.empty:
        send_email_report(f"<h2>Báo cáo ngày {target_date.strftime('%d/%m/%Y')}</h2><p>Không có thí sinh nào nhắn tin vào ngày hôm qua.</p>")
        return

    df_user_unique = df_user.sort_values('Thời gian').drop_duplicates(subset=['Email'], keep='last')
    
    # Vector Scoring để lọc nhiễu
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    query_vec = embedder.encode(["ngành điểm chuẩn xét tuyển học phí chỉ tiêu ký túc xá học bổng"])
    all_questions = df_chat['Câu hỏi'].fillna("").tolist()
    all_vecs = embedder.encode(all_questions, batch_size=128)
    df_chat['Score'] = cosine_similarity(query_vec, all_vecs)[0]
    
    chat_grouped = (
        df_chat[df_chat['Score'] > 0.2]
        .sort_values(['Email', 'Score'], ascending=[True, False])
        .groupby('Email').head(5)
        .sort_values(['Email', 'Thời gian'])
        .groupby('Email')['Câu hỏi'].apply(list).reset_index()
    )

    merged_df = pd.merge(df_user_unique, chat_grouped, on='Email', how='inner')
    
    # Kiểm tra lần cuối sau khi lọc ngữ nghĩa
    if merged_df.empty:
        send_email_report(f"<h2>Báo cáo ngày {target_date.strftime('%d/%m/%Y')}</h2><p>Không có câu hỏi hợp lệ nào cần tư vấn (các tin nhắn hôm qua chỉ là giao tiếp xã giao).</p>")
        return

    # Chạy AI tóm tắt
    print(f"Đang phân tích {len(merged_df)} thí sinh...")
    summaries = asyncio.run(process_all_summaries(merged_df['Câu hỏi'].tolist()))
    merged_df['Nội dung quan tâm'] = summaries
    final_df = merged_df[['Họ Tên', 'Số điện thoại', 'Email', 'Địa chỉ', 'Nội dung quan tâm']]

    # Cập nhật Google Cloud Sheet
    update_cloud_summary_sheet(service, final_df, target_date)

    # Gửi Email dạng bảng HTML
    html_table = final_df.to_html(classes='table', justify='center', index=False)
    html_body = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; font-family: sans-serif; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #004085; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2 style="color: #004085;">Báo cáo thí sinh ngày {target_date.strftime('%d/%m/%Y')}</h2>
        <p>Hệ thống đã tự động lưu dữ liệu vào sheet tổng hợp tháng trên Cloud.</p>
        {html_table}
        <br><p><i>-- Hệ thống báo cáo tự động BLU --</i></p>
    </body>
    </html>
    """
    send_email_report(html_body)
    print("Hoàn tất báo cáo.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Bắt toàn bộ lỗi ngoài dự kiến và in ra console/log
        print(f"Lỗi nghiêm trọng khi chạy script: {e}")
        traceback.print_exc()
        # Gọi luôn hàm gửi mail lỗi cho Quản trị viên
        send_email_report(f"<h3>LỖI HỆ THỐNG CRITICAL</h3><p><pre>{traceback.format_exc()}</pre></p>", is_error=True)