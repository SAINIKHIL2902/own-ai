// State
let chats = []; // [{ id: string, title: string, messages: [{ id, role, content, feedback }] }]
let currentChatId = null;
let isGenerating = false;
let activeModel = "llama3.2:1b";
let currentMode = localStorage.getItem("own_ai_mode") || "auto"; // "auto", "offline", "online"
let selectedModelOverride = localStorage.getItem("own_ai_selected_model") || "auto";
let cachedLocalModels = ["llama3.2:1b"];
let cachedOnlineModels = ["gemini-3.5-flash-lite"];

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

// Mode & Model Picker Elements
const modeBtnAuto = document.getElementById("mode-btn-auto");
const modeBtnOffline = document.getElementById("mode-btn-offline");
const modeBtnOnline = document.getElementById("mode-btn-online");
const modelPickerSelect = document.getElementById("model-picker-select");
const offlinePillDot = document.getElementById("offline-pill-dot");
const offlinePillText = document.getElementById("offline-pill-text");

const messagesViewport = document.getElementById("messages-viewport");
const welcomeContainer = document.getElementById("welcome-container");
const conversationStream = document.getElementById("conversation-stream");

const chatTextarea = document.getElementById("chat-textarea");
const sendButton = document.getElementById("send-button");
const errorToast = document.getElementById("error-toast");
const toastMessage = document.getElementById("toast-message");
const toastClose = document.getElementById("toast-close");

// Profile Modal Elements
const profileModal = document.getElementById("profile-modal");
const viewProfileBtn = document.getElementById("view-profile-btn");
const topProfileBtn = document.getElementById("top-profile-btn");
const modalCloseBtn = document.getElementById("modal-close-btn");
const modalInterests = document.getElementById("modal-interests");
const modalPreferences = document.getElementById("modal-preferences");
const modalResponseStyles = document.getElementById("modal-response-styles");
const modalConfusionTriggers = document.getElementById("modal-confusion-triggers");
const modalStats = document.getElementById("modal-stats");
const exportDatasetBtn = document.getElementById("export-dataset-btn");
const exportStatus = document.getElementById("export-status");

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
      appendMessageToDOM(msg.role, msg.content, false, msg.id, msg.feedback, msg.provider, msg.model);
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

// Submit Feedback Handler
window.submitFeedback = async function (msgId, type, btnElement) {
  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message_id: msgId,
        conversation_id: currentChatId,
        feedback_type: type === "positive" ? "thumbs_up" : "thumbs_down",
        feedback_value: type,
        rating: type === "positive" ? 5 : 1,
      }),
    });
    if (res.ok) {
      const parent = btnElement.closest(".msg-actions");
      parent.querySelectorAll(".feedback-btn").forEach((b) => {
        b.classList.remove("active-positive", "active-negative");
      });
      btnElement.classList.add(type === "positive" ? "active-positive" : "active-negative");

      // Record in local state
      const chat = getCurrentChat();
      if (chat) {
        const msg = chat.messages.find((m) => m.id === msgId);
        if (msg) msg.feedback = type;
        saveStorage();
      }
    }
  } catch (e) {
    console.error("Feedback submit failed:", e);
  }
};

// Append Message to DOM
function appendMessageToDOM(role, content, animate = true, msgId = null, feedback = null, provider = null, model = null) {
  welcomeContainer.style.display = "none";
  conversationStream.style.display = "flex";

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;
  const effectiveId = msgId || "msg_" + Math.random().toString(36).substr(2, 9);

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

  // Actions bar for assistant (Copy + Thumbs Up + Thumbs Down + Model Attribution)
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

    // Thumbs Up button
    const thumbsUpBtn = document.createElement("button");
    thumbsUpBtn.className = `feedback-btn ${feedback === "positive" ? "active-positive" : ""}`;
    thumbsUpBtn.innerHTML = "👍";
    thumbsUpBtn.title = "Good response";
    thumbsUpBtn.addEventListener("click", () => window.submitFeedback(effectiveId, "positive", thumbsUpBtn));
    actions.appendChild(thumbsUpBtn);

    // Thumbs Down button
    const thumbsDownBtn = document.createElement("button");
    thumbsDownBtn.className = `feedback-btn ${feedback === "negative" ? "active-negative" : ""}`;
    thumbsDownBtn.innerHTML = "👎";
    thumbsDownBtn.title = "Poor response";
    thumbsDownBtn.addEventListener("click", () => window.submitFeedback(effectiveId, "negative", thumbsDownBtn));
    actions.appendChild(thumbsDownBtn);

    // Model Attribution Badge
    const effectiveProvider = provider || "local";
    const badge = document.createElement("span");
    badge.className = `model-attribution-badge ${effectiveProvider === "gemini" ? "provider-gemini" : "provider-local"}`;
    const providerName = effectiveProvider === "gemini" ? "Gemini" : "Ollama";
    const modelTag = model ? ` (${escapeHtml(model)})` : "";
    badge.innerHTML = `Answered by: <strong>${providerName}</strong>${modelTag}`;
    actions.appendChild(badge);

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

  if (chat.messages.length === 0) {
    chat.title = text.slice(0, 30) + (text.length > 30 ? "..." : "");
    renderHistory();
  }

  const userMsgId = "msg_" + Math.random().toString(36).substr(2, 9);
  chat.messages.push({ id: userMsgId, role: "user", content: text });
  appendMessageToDOM("user", text, true, userMsgId);
  saveStorage();

  showThinking();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: currentChatId,
        messages: chat.messages.map((m) => ({ role: m.role, content: m.content })),
        mode: currentMode,
        model: selectedModelOverride !== "auto" ? selectedModelOverride : null,
      }),
    });

    removeThinking();

    if (!response.ok) {
      let errDetail = `Server error (status ${response.status})`;
      try {
        const errJson = await response.json();
        errDetail = errJson.detail || errDetail;
      } catch (e) {}
      if (errDetail.toLowerCase().includes("ollama is not running")) {
        errDetail = "Local Ollama is not running. Run 'ollama serve' in your terminal or switch to '☁️ Online' mode.";
      }
      showErrorToast(errDetail);
      chat.messages.pop();
      saveStorage();
      return;
    }

    const data = await response.json();
    const asstMsgId = data.message_id || ("msg_" + Math.random().toString(36).substr(2, 9));
    const provider = data.provider || "local";
    const respModel = data.model || activeModel;

    chat.messages.push({
      id: asstMsgId,
      role: "assistant",
      content: data.response,
      provider: provider,
      model: respModel
    });
    appendMessageToDOM("assistant", data.response, true, asstMsgId, null, provider, respModel);
    saveStorage();

    if (data.model) {
      activeModel = data.model;
      if (topModelName) topModelName.textContent = activeModel;
      if (sidebarModelBadge) sidebarModelBadge.textContent = activeModel;
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

function setRoutingMode(mode, triggerModelUpdate = true) {
  currentMode = mode;
  localStorage.setItem("own_ai_mode", mode);

  // Update button active state
  [modeBtnAuto, modeBtnOffline, modeBtnOnline].forEach((btn) => {
    if (!btn) return;
    if (btn.dataset.mode === mode) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update status pill & dropdown selection
  if (mode === "offline") {
    if (offlinePillDot) offlinePillDot.className = "dot-green";
    if (offlinePillText) offlinePillText.textContent = "100% Offline";
    if (triggerModelUpdate) {
      const isCurrentLocal = cachedLocalModels.includes(selectedModelOverride);
      selectedModelOverride = isCurrentLocal ? selectedModelOverride : (cachedLocalModels[0] || "llama3.2:1b");
      if (modelPickerSelect) modelPickerSelect.value = selectedModelOverride;
    }
  } else if (mode === "online") {
    if (offlinePillDot) offlinePillDot.className = "dot-blue";
    if (offlinePillText) offlinePillText.textContent = "Gemini Online";
    if (triggerModelUpdate) {
      const isCurrentOnline = cachedOnlineModels.includes(selectedModelOverride);
      selectedModelOverride = isCurrentOnline ? selectedModelOverride : (cachedOnlineModels[0] || "gemini-3.5-flash-lite");
      if (modelPickerSelect) modelPickerSelect.value = selectedModelOverride;
    }
  } else {
    // Auto mode
    if (offlinePillDot) offlinePillDot.className = "dot-purple";
    if (offlinePillText) offlinePillText.textContent = "Auto Hybrid";
    if (triggerModelUpdate) {
      selectedModelOverride = "auto";
      if (modelPickerSelect) modelPickerSelect.value = "auto";
    }
  }

  localStorage.setItem("own_ai_selected_model", selectedModelOverride);
}

function initModeSelectorListeners() {
  if (modeBtnAuto) modeBtnAuto.addEventListener("click", () => setRoutingMode("auto"));
  if (modeBtnOffline) modeBtnOffline.addEventListener("click", () => setRoutingMode("offline"));
  if (modeBtnOnline) modeBtnOnline.addEventListener("click", () => setRoutingMode("online"));

  if (modelPickerSelect) {
    modelPickerSelect.addEventListener("change", (e) => {
      const val = e.target.value;
      selectedModelOverride = val;
      localStorage.setItem("own_ai_selected_model", val);

      if (val === "auto") {
        setRoutingMode("auto", false);
      } else if (val.toLowerCase().startsWith("gemini")) {
        setRoutingMode("online", false);
      } else {
        setRoutingMode("offline", false);
      }
    });
  }
}

async function loadModels() {
  try {
    const res = await fetch("/models");
    if (!res.ok) return;
    const data = await res.json();

    if (data.local_models && data.local_models.length > 0) {
      cachedLocalModels = data.local_models;
    }
    if (data.online_models && data.online_models.length > 0) {
      cachedOnlineModels = data.online_models;
    }

    // Populate model dropdown
    if (modelPickerSelect) {
      const savedMode = localStorage.getItem("own_ai_mode") || "auto";
      const savedModel = localStorage.getItem("own_ai_selected_model") || "auto";

      let html = `<option value="auto">✨ Auto Router (Hybrid)</option>`;

      html += `<optgroup label="💻 Offline Local Models">`;
      cachedLocalModels.forEach((m) => {
        html += `<option value="${escapeHtml(m)}">${escapeHtml(m)} (Local)</option>`;
      });
      html += `</optgroup>`;

      html += `<optgroup label="☁️ Online Cloud Models">`;
      cachedOnlineModels.forEach((m) => {
        html += `<option value="${escapeHtml(m)}">${escapeHtml(m)} (Cloud)</option>`;
      });
      html += `</optgroup>`;

      modelPickerSelect.innerHTML = html;

      // Restore saved selection
      selectedModelOverride = savedModel;
      modelPickerSelect.value = savedModel;
      setRoutingMode(savedMode, false);
    }

    if (sidebarModelBadge) {
      sidebarModelBadge.textContent = cachedLocalModels[0] || "llama3.2:1b";
    }
  } catch (e) {
    console.warn("Could not load models list:", e);
  }
}

// Open & Populate Profile Modal
async function openProfileModal() {
  profileModal.style.display = "flex";
  exportStatus.textContent = "";

  try {
    const res = await fetch("/profile");
    if (!res.ok) throw new Error("Failed to fetch profile");
    const data = await res.json();

    // 1. Interests
    if (data.interests && data.interests.length > 0) {
      modalInterests.innerHTML = data.interests
        .map(
          (item) =>
            `<div class="interest-chip">
              <span>${escapeHtml(item.topic)}</span>
              <span class="interest-score">${Math.round(item.score * 100)}%</span>
            </div>`
        )
        .join("");
    } else {
      modalInterests.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No interests modeled yet. Start chatting to extract topics!</div>';
    }

    // 2. Preferences
    if (data.preferences && data.preferences.length > 0) {
      modalPreferences.innerHTML = data.preferences
        .map(
          (pref) =>
            `<div class="pref-row">
              <span>${escapeHtml(pref.preference.replace("prefers_", "").replace(/_/g, " "))}</span>
              <div class="pref-conf-bar">
                <div class="conf-pill">
                  <div class="conf-fill" style="width: ${Math.round(pref.confidence * 100)}%"></div>
                </div>
                <span>${Math.round(pref.confidence * 100)}% confidence</span>
              </div>
            </div>`
        )
        .join("");
    } else {
      modalPreferences.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No stylistic preferences detected yet.</div>';
    }

    // 3. Preferred Response Styles
    const styles = data.response_styles || [];
    const likedStyles = styles.filter((s) => s.liked_count > 0);
    if (likedStyles.length > 0) {
      modalResponseStyles.innerHTML = likedStyles
        .map(
          (st) =>
            `<div class="pref-row">
              <div>
                <strong>${escapeHtml(st.style_name)}</strong>
                <div style="font-size: 0.78rem; color: var(--text-muted);">${escapeHtml(st.description)}</div>
              </div>
              <div class="pref-conf-bar">
                <div class="conf-pill">
                  <div class="conf-fill" style="width: ${Math.round(st.affinity_score * 100)}%"></div>
                </div>
                <span style="color: var(--accent-green); font-size: 0.78rem;">${st.liked_count} 👍 (${Math.round(st.affinity_score * 100)}% affinity)</span>
              </div>
            </div>`
        )
        .join("");
    } else {
      modalResponseStyles.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No response styles rated yet. Upvote 👍 responses you like to build your style profile!</div>';
    }

    // 4. Disliked Styles & Confusion Triggers
    const dislikedStyles = styles.filter((s) => s.disliked_count > 0);
    const confusionTriggers = [];
    if (dislikedStyles.length > 0) {
      dislikedStyles.forEach((ds) => {
        confusionTriggers.push(`${ds.style_name} (${ds.disliked_count} 👎)`);
      });
    }
    const stats = data.behavior_stats || {};
    if (stats.top_confusion_trigger) {
      confusionTriggers.push(stats.top_confusion_trigger);
    }

    if (confusionTriggers.length > 0) {
      modalConfusionTriggers.innerHTML = confusionTriggers
        .map(
          (trig) =>
            `<div class="interest-chip" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.1);">
              <span>⚠️ ${escapeHtml(trig)}</span>
            </div>`
        )
        .join("");
    } else {
      modalConfusionTriggers.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No confusion triggers or ambiguity flagged yet. Downvote 👎 confusing responses to record what to avoid.</div>';
    }

    // 5. Statistics
    modalStats.innerHTML = `
      <div class="stat-card">
        <div class="stat-label">Total Messages</div>
        <div class="stat-val">${stats.total_interactions || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Prompt Length</div>
        <div class="stat-val">${stats.avg_prompt_length || 0} chars</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Positive Feedback</div>
        <div class="stat-val">${stats.feedback_positive_count || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Negative Feedback</div>
        <div class="stat-val">${stats.feedback_negative_count || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Top Liked Style</div>
        <div class="stat-val" style="font-size: 0.85rem;">${stats.top_liked_style || "None yet"}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Last Model Used</div>
        <div class="stat-val" style="font-size: 0.95rem;">${stats.last_model_used || "N/A"}</div>
      </div>
    `;
  } catch (err) {
    modalInterests.innerHTML = '<div style="color: var(--accent-red);">Error loading user profile.</div>';
  }
}

// Export Dataset
exportDatasetBtn.addEventListener("click", async () => {
  exportStatus.textContent = "Exporting dataset...";
  try {
    const res = await fetch("/analytics/export", { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      exportStatus.textContent = `✓ Exported ${data.records_exported} records to ${data.file_path.split("/").pop()}`;
    } else {
      exportStatus.textContent = "Export failed.";
    }
  } catch (e) {
    exportStatus.textContent = "Export failed: " + e.message;
  }
});

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
newChatBtn.addEventListener("click", createNewChat);

clearAllBtn.addEventListener("click", () => {
  if (confirm("Are you sure you want to clear all conversation history?")) {
    chats = [];
    localStorage.removeItem("own_ai_chats");
    createNewChat();
  }
});

sidebarCloseBtn.addEventListener("click", () => sidebar.classList.add("collapsed"));
sidebarOpenBtn.addEventListener("click", () => sidebar.classList.remove("collapsed"));

viewProfileBtn.addEventListener("click", openProfileModal);
topProfileBtn.addEventListener("click", openProfileModal);
modalCloseBtn.addEventListener("click", () => (profileModal.style.display = "none"));

window.addEventListener("click", (e) => {
  if (e.target === profileModal) {
    profileModal.style.display = "none";
  }
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

// ==============================================================================
// PHASE 4: MEMORY MANAGEMENT UI LOGIC
// ==============================================================================
const memoryModal = document.getElementById("memory-modal");
const topMemoryBtn = document.getElementById("top-memory-btn");
const viewMemoryBtn = document.getElementById("view-memory-btn");
const memoryModalCloseBtn = document.getElementById("memory-modal-close-btn");
const memoryCountBadge = document.getElementById("memory-count-badge");
const newMemoryContent = document.getElementById("new-memory-content");
const newMemoryType = document.getElementById("new-memory-type");
const addMemoryBtn = document.getElementById("add-memory-btn");
const memoryAddMsg = document.getElementById("memory-add-msg");
const memoryListContainer = document.getElementById("memory-list-container");

let currentMemoryFilter = "all";
let cachedMemories = [];

async function openMemoryModal() {
  memoryModal.style.display = "flex";
  await loadMemories();
}

function closeMemoryModal() {
  memoryModal.style.display = "none";
}

async function loadMemories() {
  memoryListContainer.innerHTML = '<div class="memory-loading">Loading memories...</div>';
  try {
    const res = await fetch("/memory");
    if (!res.ok) throw new Error("Failed to load memories");
    cachedMemories = await res.json();
    renderMemoryList();
  } catch (err) {
    memoryListContainer.innerHTML = `<div class="memory-empty">Error loading memories: ${err.message}</div>`;
  }
}

function renderMemoryList() {
  let filtered = cachedMemories;
  if (currentMemoryFilter !== "all") {
    filtered = cachedMemories.filter(m => m.memory_type.toLowerCase() === currentMemoryFilter.toLowerCase());
  }

  memoryCountBadge.textContent = `${cachedMemories.length} ${cachedMemories.length === 1 ? "memory" : "memories"}`;

  if (filtered.length === 0) {
    memoryListContainer.innerHTML = '<div class="memory-empty">No memories found for this filter.</div>';
    return;
  }

  memoryListContainer.innerHTML = filtered.map(m => {
    const confPercent = Math.round((m.confidence || 0.95) * 100);
    const impPercent = Math.round((m.importance || 0.8) * 100);
    const typeClass = `type-${m.memory_type.toLowerCase()}`;
    const statusClass = `status-${(m.status || "active").toLowerCase()}`;

    return `
      <div class="memory-item-card" id="mem-card-${m.memory_id}">
        <div class="memory-item-header">
          <div class="memory-tags">
            <span class="mem-type-badge ${typeClass}">${m.memory_type}</span>
            <span class="mem-meta-pill" title="Confidence">🎯 ${confPercent}%</span>
            <span class="mem-meta-pill" title="Importance">⚡ ${impPercent}%</span>
            <span class="mem-meta-pill" title="Source">${m.source_type}</span>
            <span class="mem-status-pill ${statusClass}">${m.status}</span>
          </div>
          <div class="memory-item-actions">
            <button class="mem-action-btn edit-mem-btn" data-id="${m.memory_id}" title="Edit Memory">✏️ Edit</button>
            <button class="mem-action-btn delete delete-mem-btn" data-id="${m.memory_id}" title="Archive Memory">🗑️ Archive</button>
          </div>
        </div>
        <div class="memory-content-display" id="mem-content-${m.memory_id}">${escapeHtml(m.content)}</div>
      </div>
    `;
  }).join("");

  // Attach card event listeners
  memoryListContainer.querySelectorAll(".delete-mem-btn").forEach(btn => {
    btn.addEventListener("click", () => handleDeleteMemory(btn.dataset.id));
  });

  memoryListContainer.querySelectorAll(".edit-mem-btn").forEach(btn => {
    btn.addEventListener("click", () => handleStartEditMemory(btn.dataset.id));
  });
}

async function handleAddMemory() {
  const content = newMemoryContent.value.trim();
  const type = newMemoryType.value;
  if (!content) {
    memoryAddMsg.textContent = "Please enter memory content.";
    memoryAddMsg.style.color = "#f87171";
    return;
  }

  addMemoryBtn.disabled = true;
  memoryAddMsg.textContent = "Saving memory...";
  memoryAddMsg.style.color = "var(--text-secondary)";

  try {
    const res = await fetch("/memory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: content,
        memory_type: type,
        confidence: 0.95,
        importance: 0.85,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to create memory");
    }

    newMemoryContent.value = "";
    memoryAddMsg.textContent = "✓ Memory added successfully!";
    memoryAddMsg.style.color = "#34d399";
    setTimeout(() => { memoryAddMsg.textContent = ""; }, 3000);
    await loadMemories();
  } catch (err) {
    memoryAddMsg.textContent = `Error: ${err.message}`;
    memoryAddMsg.style.color = "#f87171";
  } finally {
    addMemoryBtn.disabled = false;
  }
}

function handleStartEditMemory(id) {
  const mem = cachedMemories.find(m => m.memory_id === id);
  if (!mem) return;

  const contentEl = document.getElementById(`mem-content-${id}`);
  if (!contentEl) return;

  contentEl.innerHTML = `
    <div class="memory-edit-area">
      <input type="text" class="memory-edit-input" id="edit-input-${id}" value="${escapeHtml(mem.content)}" />
      <button class="mem-action-btn" id="save-edit-${id}" style="background:var(--accent-green);color:#fff;">Save</button>
      <button class="mem-action-btn" id="cancel-edit-${id}">Cancel</button>
    </div>
  `;

  document.getElementById(`save-edit-${id}`).addEventListener("click", async () => {
    const newText = document.getElementById(`edit-input-${id}`).value.trim();
    if (!newText) return;
    try {
      const res = await fetch(`/memory/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: newText }),
      });
      if (res.ok) {
        await loadMemories();
      }
    } catch (e) {
      alert("Failed to update memory: " + e.message);
    }
  });

  document.getElementById(`cancel-edit-${id}`).addEventListener("click", () => {
    renderMemoryList();
  });
}

async function handleDeleteMemory(id) {
  if (!confirm("Are you sure you want to archive/delete this memory?")) return;
  try {
    const res = await fetch(`/memory/${id}?soft=true`, { method: "DELETE" });
    if (res.ok) {
      await loadMemories();
    }
  } catch (e) {
    alert("Failed to delete memory: " + e.message);
  }
}

// Memory Modal Listeners
if (topMemoryBtn) topMemoryBtn.addEventListener("click", openMemoryModal);
if (viewMemoryBtn) viewMemoryBtn.addEventListener("click", openMemoryModal);
if (memoryModalCloseBtn) memoryModalCloseBtn.addEventListener("click", closeMemoryModal);

if (addMemoryBtn) addMemoryBtn.addEventListener("click", handleAddMemory);
if (newMemoryContent) {
  newMemoryContent.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddMemory();
    }
  });
}

// Memory Filter Tab Listeners
document.querySelectorAll(".memory-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".memory-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    currentMemoryFilter = tab.dataset.type;
    renderMemoryList();
  });
});

window.addEventListener("click", (e) => {
  if (e.target === memoryModal) {
    closeMemoryModal();
  }
  if (e.target === skillsModal) {
    closeSkillsModal();
  }
});

// ==============================================================================
// USER SKILLS & EXPERTISE UI LOGIC (Local SQLite + Cloud Database Sync)
// ==============================================================================
const skillsModal = document.getElementById("skills-modal");
const topSkillsBtn = document.getElementById("top-skills-btn");
const viewSkillsBtn = document.getElementById("view-skills-btn");
const composerSkillsBtn = document.getElementById("composer-skills-btn");
const skillsModalCloseBtn = document.getElementById("skills-modal-close-btn");

const skillCountBadge = document.getElementById("skill-count-badge");
const skillCloudStatus = document.getElementById("skill-cloud-status");
const tabAddSingle = document.getElementById("tab-add-single");
const tabBulkUpload = document.getElementById("tab-bulk-upload");
const skillsSingleView = document.getElementById("skills-single-view");
const skillsBulkView = document.getElementById("skills-bulk-view");

const newSkillName = document.getElementById("new-skill-name");
const newSkillLevel = document.getElementById("new-skill-level");
const newSkillCategory = document.getElementById("new-skill-category");
const newSkillYears = document.getElementById("new-skill-years");
const addSkillBtn = document.getElementById("add-skill-btn");

const skillsDropzone = document.getElementById("skills-dropzone");
const skillsFileInput = document.getElementById("skills-file-input");
const bulkSkillsText = document.getElementById("bulk-skills-text");
const bulkDefaultCategory = document.getElementById("bulk-default-category");
const bulkDefaultLevel = document.getElementById("bulk-default-level");
const bulkUploadBtn = document.getElementById("bulk-upload-btn");

const skillsFeedbackMsg = document.getElementById("skills-feedback-msg");
const skillsGridContainer = document.getElementById("skills-grid-container");
const syncSkillsCloudBtn = document.getElementById("sync-skills-cloud-btn");
const skillFooterStatus = document.getElementById("skill-footer-status");

let cachedSkills = [];
let currentSkillCategoryFilter = "all";

async function openSkillsModal() {
  if (skillsModal) {
    skillsModal.style.display = "flex";
    await loadSkills();
  }
}

function closeSkillsModal() {
  if (skillsModal) {
    skillsModal.style.display = "none";
  }
}

async function loadSkills() {
  if (!skillsGridContainer) return;
  skillsGridContainer.innerHTML = '<div class="skills-loading">Loading skills from database...</div>';
  try {
    const res = await fetch("/api/skills");
    if (!res.ok) throw new Error("Failed to load skills");
    const data = await res.json();
    cachedSkills = data.skills || [];

    if (skillCloudStatus) {
      if (data.cloud_configured) {
        skillCloudStatus.innerHTML = "☁️ Supabase Connected";
        skillCloudStatus.style.display = "inline-flex";
      } else {
        skillCloudStatus.innerHTML = "💾 Local SQLite";
        skillCloudStatus.style.display = "inline-flex";
      }
    }

    renderSkillsList();
  } catch (err) {
    skillsGridContainer.innerHTML = `<div class="skills-empty">Error loading skills: ${err.message}</div>`;
  }
}

function renderSkillsList() {
  if (!skillsGridContainer) return;

  let filtered = cachedSkills;
  if (currentSkillCategoryFilter !== "all") {
    filtered = cachedSkills.filter(
      (s) => (s.category || "").toLowerCase() === currentSkillCategoryFilter.toLowerCase()
    );
  }

  if (skillCountBadge) {
    skillCountBadge.textContent = `${cachedSkills.length} ${cachedSkills.length === 1 ? "skill" : "skills"}`;
  }

  if (filtered.length === 0) {
    skillsGridContainer.innerHTML =
      '<div class="skills-empty">No skills found in this category. Add or upload some above!</div>';
    return;
  }

  skillsGridContainer.innerHTML = filtered
    .map((s) => {
      const level = s.level || "Intermediate";
      const levelClass = `level-${level.toLowerCase()}`;
      const yearsText = s.years_experience && s.years_experience > 0 ? ` &bull; ${s.years_experience} yrs` : "";

      return `
      <div class="skill-card" id="skill-card-${encodeURIComponent(s.skill_name)}">
        <div class="skill-card-main">
          <div class="skill-card-title" title="${escapeHtml(s.skill_name)}">${escapeHtml(s.skill_name)}</div>
          <div class="skill-card-badges">
            <span class="skill-level-badge ${levelClass}">${level}</span>
            <span class="skill-category-tag">${escapeHtml(s.category || "General")}${yearsText}</span>
          </div>
        </div>
        <button class="skill-delete-btn" data-skill="${escapeHtml(s.skill_name)}" title="Remove skill from database">
          🗑️
        </button>
      </div>
    `;
    })
    .join("");

  // Attach delete listeners
  skillsGridContainer.querySelectorAll(".skill-delete-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const skillName = btn.dataset.skill;
      if (!confirm(`Remove "${skillName}" from your skills database?`)) return;
      try {
        btn.disabled = true;
        btn.textContent = "⏳";
        const res = await fetch(`/api/skills/${encodeURIComponent(skillName)}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Failed to delete skill");
        const data = await res.json();
        showSkillFeedback(`Skill "${skillName}" removed from database.`, "success");
        await loadSkills();
      } catch (err) {
        showSkillFeedback(err.message, "error");
        btn.disabled = false;
        btn.textContent = "🗑️";
      }
    });
  });
}

function showSkillFeedback(msg, type = "success") {
  if (!skillsFeedbackMsg) return;
  skillsFeedbackMsg.textContent = msg;
  skillsFeedbackMsg.className = `skill-feedback-msg ${type}`;
  setTimeout(() => {
    if (skillsFeedbackMsg.textContent === msg) {
      skillsFeedbackMsg.textContent = "";
      skillsFeedbackMsg.className = "skill-feedback-msg";
    }
  }, 4500);
}

// Subtab toggling
if (tabAddSingle && tabBulkUpload) {
  tabAddSingle.addEventListener("click", () => {
    tabAddSingle.classList.add("active");
    tabBulkUpload.classList.remove("active");
    skillsSingleView.style.display = "block";
    skillsBulkView.style.display = "none";
  });

  tabBulkUpload.addEventListener("click", () => {
    tabBulkUpload.classList.add("active");
    tabAddSingle.classList.remove("active");
    skillsSingleView.style.display = "none";
    skillsBulkView.style.display = "block";
  });
}

// Single skill addition
if (addSkillBtn) {
  addSkillBtn.addEventListener("click", async () => {
    const name = newSkillName.value.trim();
    if (!name) {
      showSkillFeedback("Please enter a skill name.", "error");
      newSkillName.focus();
      return;
    }

    addSkillBtn.disabled = true;
    addSkillBtn.textContent = "Saving to Database...";

    try {
      const res = await fetch("/api/skills", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          skill_name: name,
          level: newSkillLevel.value,
          category: newSkillCategory.value,
          years_experience: parseFloat(newSkillYears.value || 0),
        }),
      });

      if (!res.ok) throw new Error("Failed to save skill");
      const data = await res.json();

      newSkillName.value = "";
      newSkillYears.value = "";

      const cloudStatusText = data.cloud_synced ? " & Synced to Supabase Cloud" : "";
      showSkillFeedback(`✅ Saved "${name}" to database${cloudStatusText}!`, "success");
      await loadSkills();
    } catch (err) {
      showSkillFeedback(err.message, "error");
    } finally {
      addSkillBtn.disabled = false;
      addSkillBtn.textContent = "+ Save Skill to Database";
    }
  });
}

// Dropzone & File Upload
if (skillsDropzone && skillsFileInput) {
  skillsDropzone.addEventListener("click", () => skillsFileInput.click());

  skillsDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    skillsDropzone.classList.add("dragover");
  });

  skillsDropzone.addEventListener("dragleave", () => {
    skillsDropzone.classList.remove("dragover");
  });

  skillsDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    skillsDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSkillsFile(e.dataTransfer.files[0]);
    }
  });

  skillsFileInput.addEventListener("change", () => {
    if (skillsFileInput.files && skillsFileInput.files.length > 0) {
      handleSkillsFile(skillsFileInput.files[0]);
    }
  });
}

function handleSkillsFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    try {
      // Try JSON first
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        bulkUploadSkillsArray(parsed);
        return;
      }
    } catch (err) {
      // Not JSON, populate textarea
    }
    bulkSkillsText.value = text;
    showSkillFeedback(`Loaded file "${file.name}". Click "Upload & Update Database" to save.`, "success");
  };
  reader.readAsText(file);
}

async function bulkUploadSkillsArray(skillsArray) {
  if (bulkUploadBtn) {
    bulkUploadBtn.disabled = true;
    bulkUploadBtn.textContent = "Uploading to Database...";
  }
  try {
    const res = await fetch("/api/skills/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skills: skillsArray }),
    });
    if (!res.ok) throw new Error("Bulk upload failed");
    const data = await res.json();
    showSkillFeedback(`✅ Imported ${data.imported_count} skills to database!`, "success");
    await loadSkills();
  } catch (err) {
    showSkillFeedback(err.message, "error");
  } finally {
    if (bulkUploadBtn) {
      bulkUploadBtn.disabled = false;
      bulkUploadBtn.textContent = "⚡ Upload & Update Database";
    }
  }
}

// Bulk text upload
if (bulkUploadBtn) {
  bulkUploadBtn.addEventListener("click", async () => {
    const text = bulkSkillsText.value.trim();
    if (!text) {
      showSkillFeedback("Please paste or upload skills first.", "error");
      bulkSkillsText.focus();
      return;
    }

    bulkUploadBtn.disabled = true;
    bulkUploadBtn.textContent = "Uploading to Database...";

    try {
      const res = await fetch("/api/skills/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raw_text: text,
          default_category: bulkDefaultCategory.value,
          default_level: bulkDefaultLevel.value,
        }),
      });

      if (!res.ok) throw new Error("Bulk upload failed");
      const data = await res.json();

      bulkSkillsText.value = "";
      const cloudStatusText = data.cloud_synced ? " & Synced to Supabase Cloud" : "";
      showSkillFeedback(`✅ Successfully imported ${data.imported_count} skills to database${cloudStatusText}!`, "success");
      await loadSkills();
    } catch (err) {
      showSkillFeedback(err.message, "error");
    } finally {
      bulkUploadBtn.disabled = false;
      bulkUploadBtn.textContent = "⚡ Upload & Update Database";
    }
  });
}

// Category filter tabs
document.querySelectorAll(".skill-filter-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".skill-filter-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    currentSkillCategoryFilter = tab.dataset.category;
    renderSkillsList();
  });
});

// Manual Supabase cloud sync button
if (syncSkillsCloudBtn) {
  syncSkillsCloudBtn.addEventListener("click", async () => {
    syncSkillsCloudBtn.disabled = true;
    syncSkillsCloudBtn.textContent = "⏳ Syncing to Supabase...";
    skillFooterStatus.textContent = "";

    try {
      const res = await fetch("/api/skills/sync-cloud", { method: "POST" });
      const data = await res.json();
      if (data.cloud_synced) {
        skillFooterStatus.textContent = `✅ Successfully synced ${data.skills_count} skills to Supabase!`;
        skillFooterStatus.style.color = "#34d399";
      } else {
        skillFooterStatus.textContent = `⚠️ Cloud sync: ${data.message}`;
        skillFooterStatus.style.color = "#fbbf24";
      }
    } catch (err) {
      skillFooterStatus.textContent = `Error: ${err.message}`;
      skillFooterStatus.style.color = "#f87171";
    } finally {
      syncSkillsCloudBtn.disabled = false;
      syncSkillsCloudBtn.textContent = "☁️ Sync All Skills to Supabase";
    }
  });
}

// Modal open triggers
if (topSkillsBtn) topSkillsBtn.addEventListener("click", openSkillsModal);
if (viewSkillsBtn) viewSkillsBtn.addEventListener("click", openSkillsModal);
if (composerSkillsBtn) composerSkillsBtn.addEventListener("click", openSkillsModal);
if (skillsModalCloseBtn) skillsModalCloseBtn.addEventListener("click", closeSkillsModal);

// Init
initStorage();
initModeSelectorListeners();
pollHealth();
loadModels();
setInterval(pollHealth, 10000);


