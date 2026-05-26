import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import ChatMessage from "../components/ChatMessage";
import { askQuestion, listDocuments, type Document, type Message } from "../services/api";

export default function ChatPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  // Document selection from URL
  const docParam = searchParams.get("doc");
  const docsParam = searchParams.get("docs");
  const initialIds = docParam
    ? [docParam]
    : docsParam
    ? docsParam.split(",").filter(Boolean)
    : [];

  const [selectedIds, setSelectedIds] = useState<string[]>(initialIds);
  const [allDocs, setAllDocs] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listDocuments().then((docs) => setAllDocs(docs.filter((d) => d.status === "ready")));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const toggleDoc = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  const send = async () => {
    const question = input.trim();
    if (!question || loading || selectedIds.length === 0) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      sources: [],
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await askQuestion(selectedIds, question, conversationId);
      setConversationId(res.conversation_id);
      setMessages((prev) => [...prev, res.message]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const selectedDocs = allDocs.filter((d) => selectedIds.includes(d.id));

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-100">
          <button
            onClick={() => navigate("/")}
            className="text-sm text-gray-500 hover:text-gray-800 flex items-center gap-1"
          >
            ← Documents
          </button>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mt-3 mb-2">
            Select documents
          </p>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {allDocs.length === 0 && (
            <p className="text-xs text-gray-400 p-2">No documents ready yet.</p>
          )}
          {allDocs.map((doc) => (
            <button
              key={doc.id}
              onClick={() => toggleDoc(doc.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                selectedIds.includes(doc.id)
                  ? "bg-blue-50 text-blue-700 font-medium"
                  : "text-gray-600 hover:bg-gray-50"
              }`}
            >
              <span className="truncate block">{doc.filename}</span>
              <span className="text-xs text-gray-400">
                {doc.page_count} pages · {doc.chunk_count} chunks
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* Main chat area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-2">
          <h1 className="font-semibold text-gray-900">
            {selectedDocs.length === 0
              ? "Select documents to start"
              : selectedDocs.length === 1
              ? selectedDocs[0].filename
              : `${selectedDocs.length} documents selected`}
          </h1>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center text-gray-400">
              <p className="text-4xl mb-3">💬</p>
              <p className="font-medium text-gray-600">Ask anything about your documents</p>
              <p className="text-sm mt-1">
                {selectedIds.length === 0
                  ? "Select at least one document from the sidebar first."
                  : `Searching across ${selectedIds.length} document${selectedIds.length > 1 ? "s" : ""}.`}
              </p>
            </div>
          )}
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
                <span className="inline-flex gap-1">
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                </span>
              </div>
            </div>
          )}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-2 text-sm mb-4">
              {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <div className="flex gap-3 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                selectedIds.length === 0
                  ? "Select a document first…"
                  : "Ask a question… (Enter to send)"
              }
              disabled={selectedIds.length === 0 || loading}
              rows={1}
              className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-400"
              style={{ maxHeight: "120px" }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
              }}
            />
            <button
              onClick={send}
              disabled={!input.trim() || selectedIds.length === 0 || loading}
              className="bg-blue-600 text-white px-4 py-2.5 rounded-xl font-medium text-sm hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              Send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
