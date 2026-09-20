// State
let chats = []; // [{ id: string, title: string, messages: [{ role, content }] }]
let currentChatId = null;
let isGenerating = false;
let activeModel = "llama3.2:1b";

// DOM Elements
const sidebar = document.getElementById("sidebar");
const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
const sidebarOpenBtn = document.getElementById("sidebar-open-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const historyList = document.getElementById("history-list");
const clearAllBtn = document.getElementById("clear-all-btn");

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const sidebarModelBadge = document.getElementById("sidebar-model-badge");
const topModelName = document.getElementById("top-model-name");

const messagesViewport = document.getElementById("messages-viewport");
const welcomeContainer = document.getElementById("welcome-container");
const conversationStream = document.getElementById("conversation-stream");

const chatTextarea = document.getElementById("chat-textarea");
const sendButton = document.getElementById("send-button");
const errorToast = document.getElementById("error-toast");
const toastMessage = document.getElementById("toast-message");
const toastClose = document.getElementById("toast-close");

// Initialize from LocalStorage
function initStorage() {
  try {
    const saved = localStorage.getItem("own_ai_chats");
    if (saved) {
      chats = JSON.parse(saved);
    }
  } catch (e) {
    chats = [];
  }

  if (chats.length > 0) {
    currentChatId = chats[0].id;
  } else {
    createNewChat();
  }
}

function saveStorage() {
  try {
    localStorage.setItem("own_ai_chats", JSON.stringify(chats));
  } catch (e) {}
}

function getCurrentChat() {
  return chats.find((c) => c.id === currentChatId);
}

function createNewChat() {
  const newChat = {
    id: "chat_" + Date.now(),
    title: "New chat",
    messages: [],
  };
  chats.unshift(newChat);
  currentChatId = newChat.id;
  saveStorage();
  renderHistory();
  renderCurrentChat();
  chatTextarea.focus();
}

function renderHistory() {
  historyList.innerHTML = "";
  if (chats.length === 0 || (chats.length === 1 && chats[0].messages.length === 0)) {
    historyList.innerHTML = '<div class="history-empty">No recent conversations</div>';
    return;
  }

  chats.forEach((chat) => {
    if (chat.messages.length === 0 && chat.id !== currentChatId) return;

    const item = document.createElement("div");
    item.className = `history-item ${chat.id === currentChatId ? "active" : ""}`;
    item.textContent = chat.title || "Untitled chat";
    item.title = chat.title;
    item.addEventListener("click", () => {
      if (currentChatId !== chat.id) {
        currentChatId = chat.id;
        renderHistory();
        renderCurrentChat();
      }
    });
    historyList.appendChild(item);
  });
}

function renderCurrentChat() {
  const chat = getCurrentChat();
  conversationStream.innerHTML = "";

  if (!chat || chat.messages.length === 0) {
    welcomeContainer.style.display = "block";
    conversationStream.style.display = "none";
  } else {
    welcomeContainer.style.display = "none";
    conversationStream.style.display = "flex";
    chat.messages.forEach((msg) => {
      appendMessageToDOM(msg.role, msg.content, false);
    });
    scrollToBottom();
  }
}

// Markdown Parser (100% Offline & Pure JS)
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatMarkdown(rawText) {
  if (!rawText) return "";

  // 1. Code blocks with ```lang ... ```
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let formatted = rawText.replace(codeBlockRegex, (match, lang, code) => {
    const language = lang.trim() || "code";
    const escapedCode = escapeHtml(code.trimEnd());
    return `
      <div class="code-block-container">
        <div class="code-block-header">
          <span>${escapeHtml(language)}</span>
          <button class="copy-code-btn" onclick="copyCodeFromBlock(this)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy code</span>
          </button>
        </div>
        <pre class="code-block-body"><code>${escapedCode}</code></pre>
      </div>
    `;
  });

  // 2. Inline code `code`
  formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
    return `<code class="inline-code">${escapeHtml(code)}</code>`;
  });

  // 3. Bold **text**
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // 4. Paragraphs and linebreaks
  const paragraphs = formatted.split(/\n\n+/);
  return paragraphs
    .map((p) => {
      p = p.trim();
      if (!p) return "";
      if (p.startsWith("<div class=\"code-block-container\"")) return p;
      return `<p>${p.replace(/\n/g, "<br/>")}</p>`;
    })
    .join("");
}

// Copy Code Block Handler
window.copyCodeFromBlock = function (btn) {
  const pre = btn.closest(".code-block-container").querySelector("pre code");
  if (pre) {
    navigator.clipboard.writeText(pre.innerText).then(() => {
      const span = btn.querySelector("span");
      const original = span.textContent;
      span.textContent = "Copied!";
      setTimeout(() => {
        span.textContent = original;
      }, 2000);
    });
  }
};

// Copy Full Message Handler
window.copyFullMessage = function (btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    const span = btn.querySelector("span");
    const original = span.textContent;
    span.textContent = "Copied!";
    setTimeout(() => {
      span.textContent = original;
    }, 2000);
  });
};

// Append Message to DOM
function appendMessageToDOM(role, content, animate = true) {
  welcomeContainer.style.display = "none";
  conversationStream.style.display = "flex";

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  // Avatar for assistant
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5z"/>
      </svg>
    `;
    row.appendChild(avatar);
  }

  const contentWrapper = document.createElement("div");
  contentWrapper.className = "msg-content-wrapper";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (role === "user") {
    bubble.textContent = content;
  } else {
    bubble.innerHTML = formatMarkdown(content);
  }

  contentWrapper.appendChild(bubble);

  // Actions bar for assistant
  if (role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "msg-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "action-icon-btn";
    copyBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy</span>
    `;
    copyBtn.addEventListener("click", () => window.copyFullMessage(copyBtn, content));
    actions.appendChild(copyBtn);

    contentWrapper.appendChild(actions);
  }

  row.appendChild(contentWrapper);
  conversationStream.appendChild(row);

  if (animate) {
    scrollToBottom();
  }
}

// Loading Indicator
function showThinking() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.id = "thinking-indicator-row";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.innerHTML = `
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5z"/>
    </svg>
  `;

  const contentWrapper = document.createElement("div");
  contentWrapper.className = "msg-content-wrapper";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble thinking-indicator";
  bubble.innerHTML = '<div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div>';

  contentWrapper.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(contentWrapper);
  conversationStream.appendChild(row);

  scrollToBottom();
}

function removeThinking() {
  const row = document.getElementById("thinking-indicator-row");
  if (row) row.remove();
}

function scrollToBottom() {
  messagesViewport.scrollTop = messagesViewport.scrollHeight;
}

// Error Handling
function showErrorToast(msg) {
  toastMessage.textContent = msg;
  errorToast.style.display = "flex";
}

function hideErrorToast() {
  errorToast.style.display = "none";
}

// Send Message
async function handleSendMessage() {
  const text = chatTextarea.value.trim();
  if (!text || isGenerating) return;

  hideErrorToast();
  isGenerating = true;
  sendButton.disabled = true;
  chatTextarea.value = "";
  chatTextarea.style.height = "auto";

  let chat = getCurrentChat();
  if (!chat) {
    createNewChat();
    chat = getCurrentChat();
  }

  // Update title if first message
  if (chat.messages.length === 0) {
    chat.title = text.slice(0, 30) + (text.length > 30 ? "..." : "");
    renderHistory();
  }

  // Append user message
  chat.messages.push({ role: "user", content: text });
  appendMessageToDOM("user", text);
  saveStorage();

  // Show thinking
  showThinking();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chat.messages,
      }),
    });

    removeThinking();

    if (!response.ok) {
      let errDetail = `Server error (status ${response.status})`;
      try {
        const errJson = await response.json();
        errDetail = errJson.detail || errDetail;
      } catch (e) {}
      showErrorToast(errDetail);
      chat.messages.pop(); // Remove failed user message from state
      saveStorage();
      return;
    }

    const data = await response.json();
    chat.messages.push({ role: "assistant", content: data.response });
    appendMessageToDOM("assistant", data.response);
    saveStorage();

    if (data.model) {
      activeModel = data.model;
      topModelName.textContent = activeModel;
      sidebarModelBadge.textContent = activeModel;
    }
  } catch (err) {
    removeThinking();
    showErrorToast(`Connection failed: ${err.message}`);
    chat.messages.pop();
    saveStorage();
  } finally {
    isGenerating = false;
    sendButton.disabled = chatTextarea.value.trim().length === 0;
    chatTextarea.focus();
  }
}

// System Health and Models Polling
async function pollHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    if (data.ollama_connected) {
      statusDot.className = "status-dot online";
      statusText.textContent = "Ollama Online";
    } else {
      statusDot.className = "status-dot offline";
      statusText.textContent = "Ollama Offline";
    }
  } catch (err) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Backend Offline";
  }
}

async function loadModels() {
  try {
    const res = await fetch("/models");
    if (!res.ok) return;
    const data = await res.json();
    if (data.models && data.models.length > 0) {
      activeModel = data.models[0];
      topModelName.textContent = activeModel;
      sidebarModelBadge.textContent = activeModel;
    }
  } catch (e) {}
}

// Event Listeners
chatTextarea.addEventListener("input", () => {
  chatTextarea.style.height = "auto";
  chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 200) + "px";
  sendButton.disabled = chatTextarea.value.trim().length === 0 || isGenerating;
});

chatTextarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendButton.disabled) {
      handleSendMessage();
    }
  }
});

sendButton.addEventListener("click", handleSendMessage);

newChatBtn.addEventListener("click", () => {
  createNewChat();
});

clearAllBtn.addEventListener("click", () => {
  if (confirm("Are you sure you want to clear all conversation history?")) {
    chats = [];
    localStorage.removeItem("own_ai_chats");
    createNewChat();
  }
});

sidebarCloseBtn.addEventListener("click", () => {
  sidebar.classList.add("collapsed");
});

sidebarOpenBtn.addEventListener("click", () => {
  sidebar.classList.remove("collapsed");
});

toastClose.addEventListener("click", hideErrorToast);

// Suggestion cards
document.addEventListener("click", (e) => {
  const card = e.target.closest(".suggestion-card");
  if (card) {
    const prompt = card.getAttribute("data-prompt");
    if (prompt) {
      chatTextarea.value = prompt;
      chatTextarea.dispatchEvent(new Event("input"));
      handleSendMessage();
    }
  }
});

// Init
initStorage();
pollHealth();
loadModels();
setInterval(pollHealth, 10000);
