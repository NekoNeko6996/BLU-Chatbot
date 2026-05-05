// blu-widget.js
(async function () {
    // CẤU HÌNH DOMAIN MÁY CHỦ API TẠI ĐÂY
    const API_BASE_URL = "https://unstraightforward-baldly-latonia.ngrok-free.dev"; // Thay đổi khi đưa lên host thực tế
    const MAX_QUESTION_IN_SESSION_CHAT = 15;        // Số lượng câu hỏi có thể hỏi trước khi hiển thị cảnh báo tùy chọn làm mới đoạn chat

    const host = document.createElement('div');
    host.id = 'blu-chatbot-host';
    document.body.appendChild(host);
    const shadow = host.attachShadow({ mode: 'open' });

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = `${API_BASE_URL}/chat-client/blu-widget.css`;
    shadow.appendChild(link);

    const katexCss = document.createElement('link');
    katexCss.rel = 'stylesheet';
    katexCss.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    shadow.appendChild(katexCss);

    const loadScript = (src) => new Promise(resolve => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        document.head.appendChild(script);
    });

    await loadScript('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js');

    try {
        // Tải UI chính
        const htmlResponse = await fetch(`${API_BASE_URL}/chat-client/blu-widget.html`);
        if (!htmlResponse.ok) throw new Error("Không thể tải giao diện Chatbot");
        const htmlContent = await htmlResponse.text();
        
        // Tải Fragment gợi ý câu hỏi
        const fragmentResponse = await fetch(`${API_BASE_URL}/chat-client/top-chat-message-fragment.html`);
        const fragmentContent = fragmentResponse.ok ? await fragmentResponse.text() : "";
        
        const container = document.createElement('div');
        container.innerHTML = htmlContent;
        shadow.appendChild(container);

        // Gắn Fragment vào Shadow DOM ngay khi tải xong
        const fragmentBox = shadow.getElementById("top-chat-message-fragment");
        if (fragmentBox) fragmentBox.innerHTML = fragmentContent;

        const katexCss = document.createElement('link');
        katexCss.rel = 'stylesheet';
        katexCss.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
        shadow.appendChild(katexCss);

        // Tải thư viện ngoại tuyến cần thiết
        const loadScript = (src) => {
            return new Promise((resolve, reject) => {
                if (document.querySelector(`script[src="${src}"]`)) return resolve();
                const script = document.createElement('script');
                script.src = src;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        };

        await loadScript("https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js");
        await loadScript("https://cdn.jsdelivr.net/npm/marked/marked.min.js");

        // Khởi tạo toàn bộ Event Listener, truyền kèm nội dung fragment để dùng sau này
        initChatbotLogic(shadow, API_BASE_URL, fragmentContent, MAX_QUESTION_IN_SESSION_CHAT);

    } catch (error) {
        console.error("BLU Chatbot Initialize Error:", error);
    }
})();

function initChatbotLogic(shadow, API_BASE_URL, fragmentContent, MAX_QUESTION_IN_SESSION_CHAT) {
    const elements = {
        toggle: shadow.getElementById("chatToggle"),
        wrapper: shadow.getElementById("chatWrapper"),
        body: shadow.getElementById("chatBody"),
        input: shadow.getElementById("inputMessage"),
        sendBtn: shadow.getElementById("sendBtn"),
        stopBtn: shadow.getElementById("stopBtn"),
        newChatBtn: shadow.getElementById("newChatBtn"),
        fullscreenBtn: shadow.getElementById("fullscreenBtn"),
        infoForm: shadow.getElementById("userInfoForm"),
        chatContainer: shadow.getElementById("chat-container"),
        submitBtn: shadow.getElementById("submitInfo"),
        phoneInput: shadow.getElementById("phone"),
        emailInput: shadow.getElementById("email"),
        phoneError: shadow.getElementById("phoneError"),
        emailError: shadow.getElementById("emailError"),
        nameInput: shadow.getElementById("name"),       
        addressInput: shadow.getElementById("address"),
        alertBox: shadow.getElementById("alert-box"),
    };

    let abortController = null;
    let userInfo = null;
	let questionCount = 0;
    let sessionId = "web_" + Date.now();

    elements.toggle.onclick = () => {
        elements.wrapper.classList.toggle("show");
    };

    elements.fullscreenBtn.onclick = () => {
        elements.wrapper.classList.toggle("fullscreen");
        elements.fullscreenBtn.innerHTML = !elements.wrapper.classList.contains("fullscreen") ? 
        `<svg width="25px" height="25px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M3 4C3 3.44772 3.44772 3 4 3H8C8.55228 3 9 3.44772 9 4C9 4.55228 8.55228 5 8 5H6.41421L9.70711 8.29289C10.0976 8.68342 10.0976 9.31658 9.70711 9.70711C9.31658 10.0976 8.68342 10.0976 8.29289 9.70711L5 6.41421V8C5 8.55228 4.55228 9 4 9C3.44772 9 3 8.55228 3 8V4ZM16 3H20C20.5523 3 21 3.44772 21 4V8C21 8.55228 20.5523 9 20 9C19.4477 9 19 8.55228 19 8V6.41421L15.7071 9.70711C15.3166 10.0976 14.6834 10.0976 14.2929 9.70711C13.9024 9.31658 13.9024 8.68342 14.2929 8.29289L17.5858 5H16C15.4477 5 15 4.55228 15 4C15 3.44772 15.4477 3 16 3ZM9.70711 14.2929C10.0976 14.6834 10.0976 15.3166 9.70711 15.7071L6.41421 19H8C8.55228 19 9 19.4477 9 20C9 20.5523 8.55228 21 8 21H4C3.44772 21 3 20.5523 3 20V16C3 15.4477 3.44772 15 4 15C4.55228 15 5 15.4477 5 16V17.5858L8.29289 14.2929C8.68342 13.9024 9.31658 13.9024 9.70711 14.2929ZM14.2929 14.2929C14.6834 13.9024 15.3166 13.9024 15.7071 14.2929L19 17.5858V16C19 15.4477 19.4477 15 20 15C20.5523 15 21 15.4477 21 16V20C21 20.5523 20.5523 21 20 21H16C15.4477 21 15 20.5523 15 20C15 19.4477 15.4477 19 16 19H17.5858L14.2929 15.7071C13.9024 15.3166 13.9024 14.6834 14.2929 14.2929Z" fill="#FFF"/>
        </svg>` 
        : 
        `<svg width="28px" height="28px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 4V7C9 8.1 8.1 9 7 9H4" stroke="#FFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M15 20L15 17C15 15.89 15.89 15 17 15L20 15" stroke="#FFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M20 9L17 9C15.89 9 15 8.1 15 7L15 4" stroke="#FFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M4 15L7 15C8.1 15 9 15.89 9 17L9 20" stroke="#FFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>`;
    };

    // XỬ LÝ LÀM MỚI CHAT
    elements.newChatBtn.onclick = () => {
        if (confirm("Làm mới cuộc trò chuyện?")) {
            // Xóa sạch hội thoại hiện tại
            elements.body.innerHTML = '';
			
			questionCount = 0;
			sessionId = "web_" + Date.now();
            
            // Tái tạo lại thẻ Fragment để bọc danh sách
            const newFragmentBox = document.createElement('div');
            newFragmentBox.id = 'top-chat-message-fragment';
            newFragmentBox.innerHTML = fragmentContent;
            
            // Đưa lại vào giao diện
            elements.body.appendChild(newFragmentBox);

            // Điền lại lời chào nếu đã đăng nhập thành công
            if (userInfo && userInfo.name) {
                const helloP = shadow.getElementById("initHelloP");
                if (helloP) {
                    helloP.innerHTML = `Xin chào <strong>${userInfo.name}</strong>, bạn có thể hỏi tôi các thông tin về tuyển sinh.`;
                }
            }
        }
    };

    elements.phoneInput.addEventListener("input", (e) => {
        e.target.value = e.target.value.replace(/[^0-9]/g, "");
        elements.phoneError.textContent = (e.target.value.length > 0 && e.target.value.length !== 10) ? "SĐT phải có 10 số!" : "";
    });

    elements.emailInput.addEventListener("input", (e) => {
        const val = e.target.value.trim();
		const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
		elements.emailError.textContent = (val && !emailRegex.test(val)) ? "Định dạng Email không hợp lệ!" : "";
    });

    elements.submitBtn.onclick = async (e) => {
        e.preventDefault();
        const data = {
            name: elements.nameInput.value.trim(),
            email: elements.emailInput.value.trim(),
            phone: elements.phoneInput.value.trim(),
            address: elements.addressInput.value.trim()
        };

        const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        
        const isInvalidEmail = !emailRegex.test(data.email);
        const isInvalidPhone = data.phone.length !== 10;

        if (!data.name || isInvalidPhone || isInvalidEmail || !data.address) {
            // alert("Vui lòng nhập đúng và đủ thông tin!");
            elements.alertBox.innerHTML = "<div class='alert alert-danger'>Vui lòng nhập đúng và đủ thông tin!</div>";
            return;
        }

        userInfo = data; 

        try {
            elements.submitBtn.disabled = true;
            elements.submitBtn.textContent = "Đang kết nối...";

            await fetch(`${API_BASE_URL}/user_info`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(userInfo)
            });
            
            elements.infoForm.classList.add("d-none");
            elements.chatContainer.classList.remove("d-none");
            elements.chatContainer.classList.add("d-flex");
            
            const helloP = shadow.getElementById("initHelloP");
            if (helloP) {
                helloP.innerHTML = `Xin chào <strong>${userInfo.name}</strong>, bạn có thể hỏi tôi các thông tin về tuyển sinh.`;
            }
        } catch (err) {
            const alertDiv = document.createElement("div");
                alertDiv.className = "connect-error-alert";
                alertDiv.innerHTML = `
                    <div style="font-weight: 700; margin: 10px 0 10px 0;">📢 THÔNG BÁO HỆ THỐNG</div>
                    <p>Phiên tư vấn đang gặp vấn đề kết nối đến máy chủ BLU</p>
                    <p>Nếu phiên tư vấn này có thông tin quan trọng với bạn, hãy ghi chú lại sau đó hãy tải lại đoạn chat để thử khôi phục kết nối.</p>
                    <button class="btn btn-sm btnReloadAlert" style="background-color: #664d03; color: white; border: none; margin-top: 10px; font-weight: bold;">Tải lại đoạn chat ngay</button>
                `;
                elements.body.appendChild(alertDiv);
                elements.body.scrollTop = elements.body.scrollHeight;
            elements.submitBtn.disabled = false;
            elements.submitBtn.textContent = "Bắt đầu trò chuyện";
        }
    };

    async function handleSend(customText) {
        const text = (typeof customText === 'string') ? customText.trim() : elements.input.value.trim();
        if (!text || elements.sendBtn.disabled) return;

        renderMsg("user", text);
        questionCount++;
        
        if (typeof customText !== 'string') {
            elements.input.value = "";
        }
        
        toggleLoading(true);
        abortController = new AbortController();

        try {
            const response = await fetch(`${API_BASE_URL}/chat_stream`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": "QA67V0pMzb2kLlSYQe7mPyJxGlKnbWaKVya2EZNFxtF16Nt29WNYPjQrNShtD6ft"
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    chat_request: text,
                    user_info: userInfo
                }),
                signal: abortController.signal
            });

            if (!response.ok) {
                if (response.status === 429) {
                    const errData = await response.json();
                    throw new Error(errData.error);
                }
                throw new Error("Lỗi kết nối máy chủ");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = "";
            const bubble = renderMsg("bot", `
<div class="three-body">
    <div class="three-body__dot"></div>
    <div class="three-body__dot"></div>
    <div class="three-body__dot"></div>
</div>
            `).querySelector(".bubble");
            let lastRenderTime = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                fullText += decoder.decode(value, { stream: true });

                const now = Date.now();
                if (now - lastRenderTime > 50) {
                    let sanitizedText = fullText.replace(/<context>/gi, "dữ liệu").replace(/<\/context>/gi, "");
                    const rawHtml = window.marked ? marked.parse(sanitizedText) : sanitizedText;
                    bubble.innerHTML = DOMPurify.sanitize(rawHtml);
                    elements.body.scrollTop = elements.body.scrollHeight;
                    lastRenderTime = now;
                }
            }

            // ĐOẠN XỬ LÝ FORMAT CUỐI CÙNG SAU KHI STREAM XONG
            let botText = fullText.replace(/<context>/gi, "dữ liệu").replace(/<\/context>/gi, "");
            let mathBlocks = [];

            botText = botText.replace(/(\\\[|\\\\\[)([\s\S]*?)(\\\]|\\\\\])/g, (match) => {
                mathBlocks.push(match);
                return `@@MATH_BLOCK_${mathBlocks.length - 1}@@`;
            });
            botText = botText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
                mathBlocks.push(match);
                return `@@MATH_BLOCK_${mathBlocks.length - 1}@@`;
            });

            let rawHtmlFinal = window.marked ? marked.parse(botText) : botText;

            mathBlocks.forEach((block, index) => {
                let normalizedBlock = block.replace(/\\\\\[/g, '\\[').replace(/\\\\\]/g, '\\]');
                rawHtmlFinal = rawHtmlFinal.replace(`@@MATH_BLOCK_${index}@@`, normalizedBlock);
            });

            bubble.innerHTML = DOMPurify.sanitize(rawHtmlFinal);

            if (window.renderMathInElement) {
                renderMathInElement(bubble, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '\\[', right: '\\]', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\(', right: '\\)', display: false}
                    ],
                    throwOnError: false,
                    trust: true
                });
            }
            elements.body.scrollTop = elements.body.scrollHeight;
            
            // XỬ LÝ GIỚI HẠN CÂU HỎI
            if (questionCount === MAX_QUESTION_IN_SESSION_CHAT) {
                const alertDiv = document.createElement("div");
                alertDiv.className = "session-alert";
                alertDiv.innerHTML = `
                    <div style="font-weight: 700; margin: 10px 0 10px 0;">📢 THÔNG BÁO HỆ THỐNG</div>
                    <p>Phiên trò chuyện đã khá dài. Để đảm bảo AI tư vấn chính xác, BLU khuyên bạn nên làm mới cuộc trò chuyện.</p>
                    <button class="btn btn-sm btnReloadAlert" style="background-color: #664d03; color: white; border: none; margin-top: 10px; font-weight: bold;">Tải lại đoạn chat ngay</button>
                    <p style="margin-top: 10px; font-size: 0.75rem; opacity: 0.8;">(Bạn có thể bỏ qua và tiếp tục đặt câu hỏi nếu muốn)</p>
                `;
                elements.body.appendChild(alertDiv);
                elements.body.scrollTop = elements.body.scrollHeight;
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                const alertDiv = document.createElement("div");
                alertDiv.className = "connect-error-alert";
                alertDiv.innerHTML = `
                    <div style="font-weight: 700; margin: 10px 0 10px 0;">📢 THÔNG BÁO HỆ THỐNG</div>
                    <p>Phiên tư vấn đang gặp vấn đề kết nối đến máy chủ BLU</p>
                    <p>Nếu phiên tư vấn này có thông tin quan trọng với bạn, hãy ghi chú lại sau đó hãy tải lại đoạn chat để thử khôi phục kết nối.</p>
                    <button class="btn btn-sm btnReloadAlert" style="background-color: #664d03; color: white; border: none; margin-top: 10px; font-weight: bold;">Tải lại đoạn chat ngay</button>
                `;
                elements.body.appendChild(alertDiv);
                elements.body.scrollTop = elements.body.scrollHeight;
            }
        } finally {
            toggleLoading(false);
        }
    }

    elements.body.addEventListener('click', (e) => {
        // Kiểm tra xem phần tử được click có class 'btnReloadAlert' không
        if (e.target && e.target.classList.contains('btnReloadAlert')) {
            const confirmMsg = "⚠️ LƯU Ý QUAN TRỌNG:\n- Toàn bộ lịch sử chat sẽ bị XÓA SẠCH.\n- Hãy ghi chú (note) hoặc copy lại các thông tin cần thiết trước khi tiếp tục.\n\nBạn có chắc chắn muốn làm mới không?";
            if (confirm(confirmMsg)) {
                elements.newChatBtn.click();
            }
        }
    });

    // Tạo hàm toàn cục để dùng cho các nút bấm
    window.sendBluMessage = function(text) {
        if (!elements.wrapper.classList.contains("show")) {
            elements.wrapper.classList.add("show");
        }
        handleSend(text);
    };

    function renderMsg(type, text) {
        const row = document.createElement("div");
        row.className = `msg ${type}`;
        
        if (type === "user") {
            const bubble = document.createElement("div");
            bubble.className = "bubble user";
            bubble.textContent = text; 
            row.appendChild(bubble);
            row.innerHTML += `<div class="avatar">👤</div>`;
        } else {
            row.innerHTML = `<div class="avatar bg-primary text-white">B</div>`;
            const bubble = document.createElement("div");
            bubble.className = "bubble bot";
            bubble.innerHTML = text === "..." ? `<span class="typing">...</span>` : text;
            row.appendChild(bubble);
        }
        
        elements.body.appendChild(row);
        elements.body.scrollTop = elements.body.scrollHeight;
        return row;
    }

    function toggleLoading(isLoading) {
        elements.sendBtn.classList.toggle("d-none", isLoading);
        elements.stopBtn.classList.toggle("d-none", !isLoading);
        elements.sendBtn.disabled = isLoading;
        if (!isLoading) abortController = null;
    }

    elements.sendBtn.onclick = handleSend;
    elements.input.onkeydown = (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleSend();
        }
    };
    elements.stopBtn.onclick = () => abortController?.abort();
}