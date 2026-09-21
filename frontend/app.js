// State
let chats = []; // [{ id: string, title: string, messages: [{ id, role, content, feedback }] }]
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
      appendMessageToDOM(msg.role, msg.content, false, msg.id, msg.feedback);
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
function appendMessageToDOM(role, content, animate = true, msgId = null, feedback = null) {
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

  // Actions bar for assistant (Copy + Thumbs Up + Thumbs Down)
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
        messages: chat.messages.map((m) => ({ role: m.role, content: m.content })),
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
      chat.messages.pop();
      saveStorage();
      return;
    }

    const data = await response.json();
    const asstMsgId = data.message_id || ("msg_" + Math.random().toString(36).substr(2, 9));
    chat.messages.push({ id: asstMsgId, role: "assistant", content: data.response });
    appendMessageToDOM("assistant", data.response, true, asstMsgId);
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

// Init
initStorage();
pollHealth();
loadModels();
setInterval(pollHealth, 10000);
