/**
 * RAG System - Frontend
 *
 * Handles:
 *   - SSE streaming Q&A
 *   - Chat session management
 *   - Theme toggle (dark/light)
 *   - Sidebar toggle
 *   - Citation rendering
 */

(function () {
    'use strict';

    // ============================================================
    // DOM
    // ============================================================
    const $ = (id) => document.getElementById(id);

    const messagesEl      = $('messages');
    const queryInput      = $('queryInput');
    const sendBtn         = $('sendBtn');
    const charCount       = $('charCount');
    const statusIndicator = $('statusIndicator');
    const chatList        = $('chatList');
    const chatTitle       = $('chatTitle');
    const sidebar         = $('sidebar');
    const sidebarBackdrop = $('sidebarBackdrop');
    const openSidebarBtn  = $('openSidebarBtn');
    const closeSidebarBtn = $('closeSidebarBtn');
    const newChatBtn      = $('newChatBtn');
    const themeToggle     = $('themeToggle');
    const iconSun         = $('iconSun');
    const iconMoon        = $('iconMoon');

    // ============================================================
    // State
    // ============================================================
    const state = {
        currentSessionId: null,
        isStreaming: false,
        isDeleting: false,
    };

    // ============================================================
    // API
    // ============================================================
    const API = {
        askStream: (query, sessionId) => {
            let url = `/ask/stream?query_text=${encodeURIComponent(query)}`;
            if (sessionId) url += `&session_id=${encodeURIComponent(sessionId)}`;
            return fetch(url);
        },
        createChat: (title = 'New Chat') =>
            fetch(`/chat/new?title=${encodeURIComponent(title)}`, { method: 'POST' })
                .then(r => r.json()),
        listChats: () => fetch('/chat/list').then(r => r.json()),
        getHistory: (id, limit = 50) =>
            fetch(`/chat/${id}/history?limit=${limit}`).then(r => r.json()),
        deleteChat: (id) =>
            fetch(`/chat/${id}`, { method: 'DELETE' }).then(r => r.json()),
        updateTitle: (id, title) =>
            fetch(`/chat/${id}/title?title=${encodeURIComponent(title)}`, { method: 'PUT' })
                .then(r => r.json()),
    };

    // ============================================================
    // Utils
    // ============================================================
    const esc = (s) => {
        const d = document.createElement('div');
        d.textContent = s ?? '';
        return d.innerHTML;
    };

    const scrollToBottom = () => {
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
    };

    const setStatus = (kind, text) => {
        const colors = {
            ready:    'bg-emerald-500',
            streaming:'bg-amber-500',
            error:    'bg-rose-500',
        };
        statusIndicator.innerHTML = `
            <span class="w-1.5 h-1.5 rounded-full ${colors[kind] || colors.ready}"></span>
            <span>${esc(text)}</span>
        `;
    };

    // ============================================================
    // Markdown formatter
    // ============================================================
    function formatAnswer(text) {
        if (!text) return '';

        // Strip inline citation markers
        let s = text
            .replace(/\[[^\]]*Document[^\]]*\]/gi, '')
            .replace(/\[[^\]]*p\.[^\]]*\]/gi, '')
            .replace(/\[general knowledge\]/gi, '')
            .replace(/\[\s*\]/g, '');

        // Escape HTML
        s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Force paragraph breaks before headers if stuck to text
        s = s.replace(/([^\n])(#{2,3}\s)/g, '$1\n\n$2');

        // Force paragraph break before **Bold Headers:**
        s = s.replace(/([^\n])(\*\*[^*]{3,80}?\*\*:)/g, '$1\n\n$2');

        // Force newline before bullet points if stuck
        s = s.replace(/([^\n])(\s+[-*]\s)/g, '$1\n$2');

        // Force newline before numbered items if stuck
        s = s.replace(/([^\n])(\s+\d+\.\s)/g, '$1\n$2');

        // Markdown -> HTML
        s = s.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
        s = s.replace(/^##\s+(.+)$/gm,  '<h2>$1</h2>');
        s = s.replace(/^#\s+(.+)$/gm,   '<h1>$1</h1>');

        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Lists
        s = s.replace(/^[\s]*[-*]\s+(.+)$/gm, '<li>$1</li>');
        s = s.replace(/^[\s]*\d+\.\s+(.+)$/gm, '<li>$1</li>');
        s = s.replace(/(<li>[\s\S]*?<\/li>\n?)+/g, (m) => '<ul>' + m + '</ul>');

        // Paragraph breaks
        s = s.replace(/\n{2,}/g, '</p><p>');
        s = s.replace(/\n/g, '<br>');

        // Cleanup
        return '<p>' + s + '</p>'
            .replace(/<p><\/p>/g, '')
            .replace(/<p>(<(?:ul|h\d|pre)>)/g, '$1')
            .replace(/(<\/(?:ul|h\d|pre)>)<\/p>/g, '$1')
            .replace(/<p>\s*<\/p>/g, '');
    }

    // ============================================================
    // Citations
    // ============================================================
    function renderCitations(citations) {
        if (!citations || !citations.length) return '';

        const real = citations.filter(c => c.document_name && c.document_name !== 'Knowledge Base');
        const list = real.length ? real : citations;
        if (!list.length) return '';

        const seen = new Set();
        const unique = [];
        for (const c of list) {
            const key = `${c.document_name}_${c.page ?? ''}`;
            if (!seen.has(key)) {
                seen.add(key);
                unique.push(c);
            }
        }

        const rows = unique.map(c => {
            const doc  = esc(c.document_name || 'Unknown');
            const page = c.page
                ? `<span class="inline-flex items-center rounded-md bg-brand-50 dark:bg-brand-900/40 text-brand-700 dark:text-brand-300 text-[11px] font-medium px-2 py-0.5">p. ${esc(String(c.page))}</span>`
                : '';
            const sec  = c.section
                ? `<span class="text-slate-400 dark:text-slate-500 text-xs">${esc(c.section)}</span>`
                : '';
            return `
                <div class="citation-card flex items-center flex-wrap gap-2 rounded-lg bg-slate-50 dark:bg-slate-800/60 px-3 py-2 border border-slate-200 dark:border-slate-700">
                    <svg class="w-3.5 h-3.5 text-brand-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                    </svg>
                    <span class="text-sm text-slate-700 dark:text-slate-200">${doc}</span>
                    ${page}
                    ${sec}
                </div>
            `;
        }).join('');

        return `
            <div class="mt-4 pt-3 border-t border-slate-200 dark:border-slate-700">
                <div class="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                    Sources
                </div>
                <div class="space-y-1.5">${rows}</div>
            </div>
        `;
    }

    // ============================================================
    // Message rendering
    // ============================================================
    function addMessage({ role, content, citations }) {
        const isUser = role === 'user';
        const wrapper = document.createElement('div');
        wrapper.className = `message flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`;

        const avatar = `
            <div class="shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold
                        ${isUser
                            ? 'bg-brand-600 text-white'
                            : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200'}">
                ${isUser ? 'U' : 'AI'}
            </div>
        `;

        const body = isUser
            ? `<div class="max-w-[85%] rounded-2xl rounded-tr-md bg-brand-600 text-white px-4 py-2.5 text-sm leading-relaxed shadow-sm">${esc(content)}</div>`
            : `<div class="max-w-[85%] rounded-2xl rounded-tl-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-3 text-sm leading-relaxed shadow-sm message-content">
                    ${formatAnswer(content)}
                    ${renderCitations(citations)}
               </div>`;

        wrapper.innerHTML = avatar + body;
        messagesEl.appendChild(wrapper);
        scrollToBottom();
        return wrapper;
    }

    function showWelcome() {
        messagesEl.innerHTML = '';
        const el = document.createElement('div');
        el.className = 'message max-w-2xl mx-auto text-center py-12';
        el.innerHTML = `
            <div class="w-14 h-14 mx-auto mb-5 rounded-2xl bg-brand-600 flex items-center justify-center text-white text-lg font-bold shadow-lg">
                RS
            </div>
            <h2 class="text-xl font-semibold mb-2">RAG System</h2>
            <p class="text-slate-500 dark:text-slate-400 text-sm mb-8">
                Ask a question about your documents.
            </p>
            <div class="flex flex-wrap gap-2 justify-center">
                <button class="example-tag rounded-full bg-slate-100 dark:bg-slate-800 hover:bg-brand-600 hover:text-white transition px-4 py-2 text-sm">
                    What are the main hazards?
                </button>
                <button class="example-tag rounded-full bg-slate-100 dark:bg-slate-800 hover:bg-brand-600 hover:text-white transition px-4 py-2 text-sm">
                    What are the critical control points?
                </button>
                <button class="example-tag rounded-full bg-slate-100 dark:bg-slate-800 hover:bg-brand-600 hover:text-white transition px-4 py-2 text-sm">
                    How to prevent contamination?
                </button>
            </div>
        `;
        messagesEl.appendChild(el);

        el.querySelectorAll('.example-tag').forEach(b => {
            b.addEventListener('click', () => {
                queryInput.value = b.textContent.trim();
                updateCharCount();
                submitQuery();
            });
        });
    }

    function addTyping() {
        removeTyping();
        const el = document.createElement('div');
        el.id = 'typingIndicator';
        el.className = 'message flex gap-3';
        el.innerHTML = `
            <div class="shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200">AI</div>
            <div class="max-w-[85%] rounded-2xl rounded-tl-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-3 text-sm shadow-sm">
                <div class="typing-dots text-slate-400">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        messagesEl.appendChild(el);
        scrollToBottom();
    }

    function removeTyping() {
        document.getElementById('typingIndicator')?.remove();
    }

    function updateStreamingContent(text) {
        const el = document.getElementById('typingIndicator');
        if (!el) return;
        const body = el.querySelector('.message-content') || el.querySelector('div > div:last-child');
        if (body) body.innerHTML = formatAnswer(text);
    }

    // ============================================================
    // Chat list
    // ============================================================
    async function loadChatList() {
        try {
            const chats = await API.listChats();
            chatList.innerHTML = '';
            if (!chats.length) {
                chatList.innerHTML = '<div class="text-center text-slate-400 dark:text-slate-600 py-8 text-xs">No conversations yet</div>';
                return;
            }
            for (const c of chats) {
                const item = document.createElement('div');
                item.className = `group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition ${
                    c.session_id === state.currentSessionId
                        ? 'bg-brand-50 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300'
                        : 'hover:bg-slate-100 dark:hover:bg-slate-800'
                }`;
                item.innerHTML = `
                    <svg class="w-4 h-4 shrink-0 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                    </svg>
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-medium truncate">${esc(c.title || 'New Chat')}</div>
                        <div class="text-[11px] opacity-60">${c.message_count || 0} messages</div>
                    </div>
                    <button class="delete-chat opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-rose-100 dark:hover:bg-rose-900/30 text-rose-600 transition" title="Delete">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                `;
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.delete-chat')) return;
                    loadChat(c.session_id);
                });
                item.querySelector('.delete-chat').addEventListener('click', (e) => {
                    e.stopPropagation();
                    deleteChat(c.session_id);
                });
                chatList.appendChild(item);
            }
        } catch (err) {
            console.error('loadChatList', err);
            chatList.innerHTML = '<div class="text-center text-rose-500 py-4 text-xs">Failed to load conversations</div>';
        }
    }

    async function loadChat(id) {
        if (state.isStreaming || state.isDeleting) return;
        try {
            state.currentSessionId = id;
            messagesEl.innerHTML = '';
            const history = await API.getHistory(id, 50);
            for (const m of history) {
                addMessage({ role: m.role, content: m.content, citations: [] });
            }
            const chats = await API.listChats();
            const found = chats.find(c => c.session_id === id);
            chatTitle.textContent = found?.title || 'New Chat';
            await loadChatList();
            scrollToBottom();
            closeSidebarOnMobile();
        } catch (err) {
            console.error('loadChat', err);
        }
    }

    async function createNewChat() {
        if (state.isStreaming) return;
        try {
            const r = await API.createChat('New Chat');
            state.currentSessionId = r.session_id;
            chatTitle.textContent = 'New Chat';
            showWelcome();
            await loadChatList();
            queryInput.focus();
            closeSidebarOnMobile();
        } catch (err) {
            console.error('createNewChat', err);
        }
    }

    async function deleteChat(id) {
        if (state.isStreaming || state.isDeleting) return;
        if (!confirm('Delete this conversation?')) return;

        state.isDeleting = true;
        try {
            await API.deleteChat(id);
            const chats = await API.listChats();
            if (!chats.length) {
                await createNewChat();
            } else if (id === state.currentSessionId) {
                await loadChat(chats[0].session_id);
            } else {
                await loadChatList();
            }
        } catch (err) {
            console.error('deleteChat', err);
        } finally {
            state.isDeleting = false;
        }
    }

    // ============================================================
    // Submit query (streaming)
    // ============================================================
    async function submitQuery() {
        const text = queryInput.value.trim();
        if (!text || state.isStreaming) return;

        if (!state.currentSessionId) {
            await createNewChat();
        }

        state.isStreaming = true;
        sendBtn.disabled = true;
        setStatus('streaming', 'Processing...');

        addMessage({ role: 'user', content: text });
        queryInput.value = '';
        updateCharCount();
        addTyping();

        let fullText = '';
        let citations = [];

        try {
            const res = await API.askStream(text, state.currentSessionId);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const raw = line.slice(6);

                    try {
                        const parsed = JSON.parse(raw);

                        if (parsed.session_id && !state.currentSessionId) {
                            state.currentSessionId = parsed.session_id;
                            await loadChatList();
                            continue;
                        }

                        if (parsed.done) {
                            citations = parsed.citations || [];
                            if (parsed.session_id) state.currentSessionId = parsed.session_id;
                            continue;
                        }

                        if (parsed.error) throw new Error(parsed.error);

                        if (parsed.token !== undefined) {
                            fullText += parsed.token;
                            updateStreamingContent(fullText);
                        }
                    } catch (e) {
                        fullText += raw;
                        updateStreamingContent(fullText);
                    }
                }
            }

            removeTyping();
            if (fullText) {
                addMessage({ role: 'assistant', content: fullText, citations });
            }
            await loadChatList();
            setStatus('ready', 'Ready');
        } catch (err) {
            removeTyping();
            addMessage({ role: 'assistant', content: `Error: ${err.message}` });
            setStatus('error', 'Error');
        } finally {
            state.isStreaming = false;
            sendBtn.disabled = queryInput.value.length === 0;
        }
    }

    // ============================================================
    // Input
    // ============================================================
    function updateCharCount() {
        const len = queryInput.value.length;
        charCount.textContent = `${len} / 10000`;
        sendBtn.disabled = len === 0 || state.isStreaming;
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + 'px';
    }

    // ============================================================
    // Sidebar
    // ============================================================
    function openSidebar() {
        sidebar.classList.remove('-translate-x-full');
        if (window.innerWidth < 768) {
            sidebarBackdrop.classList.remove('hidden');
        }
    }

    function closeSidebar() {
        sidebar.classList.add('-translate-x-full');
        sidebarBackdrop.classList.add('hidden');
    }

    function closeSidebarOnMobile() {
        if (window.innerWidth < 768) {
            closeSidebar();
        }
    }

    // ============================================================
    // Theme
    // ============================================================
    function applyTheme(theme) {
        const isDark = theme === 'dark';
        document.documentElement.classList.toggle('dark', isDark);
        iconSun.classList.toggle('hidden', !isDark);
        iconMoon.classList.toggle('hidden', isDark);
        localStorage.setItem('theme', theme);
    }

    function initTheme() {
        const saved = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        applyTheme(saved || (prefersDark ? 'dark' : 'light'));
    }

    // ============================================================
    // Init
    // ============================================================
    async function init() {
        initTheme();

        // If mobile, start with sidebar closed
        if (window.innerWidth < 768) {
            closeSidebar();
        }

        // Theme toggle
        themeToggle.addEventListener('click', () => {
            const isDark = document.documentElement.classList.contains('dark');
            applyTheme(isDark ? 'light' : 'dark');
        });

        // Input
        queryInput.addEventListener('input', updateCharCount);
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuery();
            }
        });
        sendBtn.addEventListener('click', submitQuery);

        // New chat
        newChatBtn.addEventListener('click', createNewChat);

        // Sidebar toggles
        openSidebarBtn.addEventListener('click', openSidebar);
        closeSidebarBtn.addEventListener('click', closeSidebar);
        sidebarBackdrop.addEventListener('click', closeSidebar);

        // ESC closes sidebar
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeSidebar();
            }
        });

        // Auto-close sidebar on resize to mobile
        window.addEventListener('resize', () => {
            if (window.innerWidth < 768) {
                closeSidebar();
            } else {
                sidebarBackdrop.classList.add('hidden');
            }
        });

        showWelcome();
        await loadChatList();

        const chats = await API.listChats();
        if (chats.length) {
            await loadChat(chats[0].session_id);
        } else {
            await createNewChat();
        }

        updateCharCount();
        setStatus('ready', 'Ready');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
