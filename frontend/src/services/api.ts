const BASE = "";

export interface Document {
  id: string;
  filename: string;
  status: "processing" | "ready" | "failed";
  page_count: number | null;
  chunk_count: number | null;
  error_msg: string | null;
  created_at: string;
}

export interface Source {
  chunk_id: string;
  page_number: number;
  content: string;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  created_at: string;
}

export interface AskResponse {
  conversation_id: string;
  message: Message;
}

// ─── Documents ───────────────────────────────────────────────────────────────

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Upload failed");
  return res.json();
}

export async function listDocuments(): Promise<Document[]> {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  const data = await res.json();
  return data.documents;
}

export async function getDocument(id: string): Promise<Document> {
  const res = await fetch(`${BASE}/documents/${id}`);
  if (!res.ok) throw new Error("Document not found");
  return res.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Delete failed");
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

export async function askQuestion(
  documentIds: string[],
  question: string,
  conversationId?: string,
): Promise<AskResponse> {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_ids: documentIds,
      question,
      conversation_id: conversationId ?? null,
    }),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed");
  return res.json();
}
