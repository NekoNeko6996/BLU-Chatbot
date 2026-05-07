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
    let latestFullTextResponse = null;
    let prevBotFooter = null;

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

        let botRow = null;

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
            let lastRenderTime = 0;
			
            if(prevBotFooter) {
                prevBotFooter.style.opacity = "0";
                setTimeout(() => { prevBotFooter.style.display = "none"; }, 300);
            }
			botRow = renderMsg("bot", "");

			const bubble = botRow.bubbleElement;
            bubble.innerHTML = `
<div class="three-body">
    <div class="three-body__dot"></div>
    <div class="three-body__dot"></div>
    <div class="three-body__dot"></div>
</div>`;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                fullText += decoder.decode(value, { stream: true });

                // 1. Tách Status cuối cùng ra
                let currentStatus = "";
                const statusRegex = /\[\[STATUS:(.*?)\]\]/g;
                let match;
                while ((match = statusRegex.exec(fullText)) !== null) {
                    currentStatus = match[1]; 
                }
                
                // 2.  Xóa tag hoàn chỉnh VÀ tag đang bị cắt dở ở cuối chuỗi do stream
                let displayText = fullText.replace(/\[\[STATUS:.*?\]\]/g, ""); // Xóa tag trọn vẹn
                displayText = displayText.replace(/\[\[STATUS:[^\]]*$/, "");   // Xóa tag bị đứt ngang (VD: "[[STATUS:Đang...")

                // 3. Cập nhật thẻ Status UI
                if (currentStatus && botRow.statusElement) {
                    botRow.statusElement.style.display = "inline-flex";
                    botRow.statusElement.querySelector("span").textContent = currentStatus;
                }

                // 4. Render text an toàn
                const now = Date.now();
                if (now - lastRenderTime > 50) {
                    let sanitizedText = displayText.replace(/<context>/gi, "dữ liệu").replace(/<\/context>/gi, "");
                    const rawHtml = window.marked ? marked.parse(sanitizedText) : sanitizedText;
                    bubble.innerHTML = DOMPurify.sanitize(rawHtml);
                    elements.body.scrollTop = elements.body.scrollHeight;
                    lastRenderTime = now;
                }
            }


            // ĐOẠN XỬ LÝ FORMAT CUỐI CÙNG SAU KHI STREAM XONG
            let botText = fullText.replace(/\[\[STATUS:[\s\S]*?\]\]/g, "")
                                  .replace(/\[\[STATUS:[^\]]*$/, "")
                                  .replace(/<context>/gi, "dữ liệu")
                                  .replace(/<\/context>/gi, "");


            let mathBlocks = [];

            botText = botText.replace(/(\\\[|\\\\\[)([\s\S]*?)(\\\]|\\\\\])/g, (match) => {
                mathBlocks.push(match);
                return `@@MATH_BLOCK_${mathBlocks.length - 1}@@`;
            });
            botText = botText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
                mathBlocks.push(match);
                return `@@MATH_BLOCK_${mathBlocks.length - 1}@@`;
            });

            latestFullTextResponse = botText;

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
            if (botRow) {
                if (botRow.statusElement) {
                    botRow.statusElement.style.opacity = "0";
                    setTimeout(() => { botRow.statusElement.style.display = "none"; }, 300);
                }

                // element button copy, like, dislike, time stamp
                if (botRow.footerElement) {
                     botRow.footerElement.style.opacity = "1";
                     setTimeout(() => { botRow.footerElement.style.display = "inline-flex"; }, 300);
                }

                prevBotFooter = botRow.footerElement;
            }
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

    window.copyToClipboard = function(element) {
        navigator.clipboard.writeText(latestFullTextResponse)
            .then(() => {
                console.log("Đã copy thành công!");
                // Có thể thêm code đổi icon thành dấu check tại đây sau...
            })
            .catch(err => {
                console.error("Lỗi copy: ", err);
            });
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
            
            // 1. Tạo Wrapper
            const wrapper = document.createElement("div");
            wrapper.className = "bot-content-wrapper";

            // 2. Tạo thẻ Badge trạng thái nổi phía trên
            const statusDiv = document.createElement("div");
            statusDiv.className = "status-badge";
            // Thêm icon loading xoay xoay
            statusDiv.innerHTML = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2V6M12 18V22M6 12H2M22 12H18M19.0711 4.92893L16.2426 7.75736M7.75736 16.2426L4.92893 19.0711M19.0711 19.0711L16.2426 16.2426M7.75736 7.75736L4.92893 4.92893" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg> <span>Đang khởi tạo...</span>`;
            statusDiv.style.display = "none"; // Ẩn lúc đầu

            // 3. Tạo Bubble Chat
            const bubble = document.createElement("div");
            bubble.className = "bubble bot";
            bubble.innerHTML = text;

            // 4. Tạo footer chứa copy, like, dislike, time
            const footer = document.createElement("div");
            footer.style.opacity = "0";
            footer.style.display = "none";
            footer.className = "bot-footer";
            footer.innerHTML = `
<button class="footer-action-btn" onclick="copyToClipboard(this)">
    <svg width="26px" height="26px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M19.5 16.5L19.5 4.5L18.75 3.75H9L8.25 4.5L8.25 7.5L5.25 7.5L4.5 8.25V20.25L5.25 21H15L15.75 20.25V17.25H18.75L19.5 16.5ZM15.75 15.75L15.75 8.25L15 7.5L9.75 7.5V5.25L18 5.25V15.75H15.75ZM6 9L14.25 9L14.25 19.5L6 19.5L6 9Z" fill="#696969"/>
    </svg>
</button>
<!--
<button class="footer-action-btn">
    <svg width="30px" height="30px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M15.0501 7.04419C15.4673 5.79254 14.5357 4.5 13.2163 4.5C12.5921 4.5 12.0062 4.80147 11.6434 5.30944L8.47155 9.75H5.85748L5.10748 10.5V18L5.85748 18.75H16.8211L19.1247 14.1428C19.8088 12.7747 19.5406 11.1224 18.4591 10.0408C17.7926 9.37439 16.8888 9 15.9463 9H14.3981L15.0501 7.04419ZM9.60751 10.7404L12.864 6.1813C12.9453 6.06753 13.0765 6 13.2163 6C13.5118 6 13.7205 6.28951 13.627 6.56984L12.317 10.5H15.9463C16.491 10.5 17.0133 10.7164 17.3984 11.1015C18.0235 11.7265 18.1784 12.6814 17.7831 13.472L15.8941 17.25H9.60751V10.7404ZM8.10751 17.25H6.60748V11.25H8.10751V17.25Z" fill="#696969"/>
    </svg>
</button>
<button class="footer-action-btn">
    <svg width="30px" height="30px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M15.0501 16.9558C15.4673 18.2075 14.5357 19.5 13.2164 19.5C12.5921 19.5 12.0063 19.1985 11.6435 18.6906L8.47164 14.25L5.85761 14.25L5.10761 13.5L5.10761 6L5.85761 5.25L16.8211 5.25L19.1247 9.85722C19.8088 11.2253 19.5407 12.8776 18.4591 13.9592C17.7927 14.6256 16.8888 15 15.9463 15L14.3982 15L15.0501 16.9558ZM9.60761 13.2596L12.8641 17.8187C12.9453 17.9325 13.0765 18 13.2164 18C13.5119 18 13.7205 17.7105 13.6271 17.4302L12.317 13.5L15.9463 13.5C16.491 13.5 17.0133 13.2836 17.3984 12.8985C18.0235 12.2735 18.1784 11.3186 17.7831 10.528L15.8941 6.75L9.60761 6.75L9.60761 13.2596ZM8.10761 6.75L6.60761 6.75L6.60761 12.75L8.10761 12.75L8.10761 6.75Z" fill="#696969"/>
    </svg>
</button>
-->
<p>${new Date()}</p>
            `;
            
            wrapper.appendChild(statusDiv);
            wrapper.appendChild(bubble);
            wrapper.appendChild(footer);
            row.appendChild(wrapper);

            row.statusElement = statusDiv;
            row.bubbleElement = bubble;
            row.footerElement = footer;
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