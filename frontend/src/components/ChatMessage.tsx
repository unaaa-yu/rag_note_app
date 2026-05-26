import ReactMarkdown from "react-markdown";
import type { Message } from "../services/api";

interface Props {
  message: Message;
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-white border border-gray-200 text-gray-800 shadow-sm"
        }`}
      >
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <>
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                code: ({ children }) => (
                  <code className="bg-gray-100 px-1 py-0.5 rounded text-xs font-mono">
                    {children}
                  </code>
                ),
                pre: ({ children }) => (
                  <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto my-2">
                    {children}
                  </pre>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>

            {message.sources.length > 0 && (
              <details className="mt-3 border-t border-gray-200 pt-2">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 select-none">
                  {message.sources.length} source{message.sources.length > 1 ? "s" : ""} used
                </summary>
                <ul className="mt-2 space-y-2">
                  {message.sources.map((src) => (
                    <li key={src.chunk_id} className="text-xs text-gray-600 bg-gray-50 rounded-lg p-2">
                      <span className="font-medium text-gray-800">Page {src.page_number}</span>
                      <span className="ml-2 text-gray-400">
                        score: {(src.score * 100).toFixed(0)}%
                      </span>
                      <p className="mt-1 text-gray-500 line-clamp-3">{src.content}</p>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </>
        )}
      </div>
    </div>
  );
}
