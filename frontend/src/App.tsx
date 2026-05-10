import {
  Activity,
  CheckCircle2,
  Clock3,
  Database,
  History,
  KeyRound,
  Loader2,
  LogOut,
  MessageSquare,
  Play,
  RefreshCw,
  RotateCw,
  Send,
  Server,
  Settings,
  Shield,
  Square,
  Trash2,
  Upload,
  Video,
} from "lucide-react";
import { FormEvent, ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

const defaultModel = "google/gemma-3-12b-it-qat-q4_0-unquantized";
const videoModes = [
  "text_to_video",
  "image_to_video",
  "video_to_video",
  "audio_to_video",
  "keyframe_interpolation",
  "retake",
  "distilled",
  "hdr",
];

type Config = {
  apiBase: string;
  token: string;
  adminKey: string;
};

type User = {
  user_id: string;
  email: string;
  created_at: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type VideoJob = {
  job_id: string;
  status: string;
  progress?: number;
  result_url?: string | null;
  error?: string | null;
  created_at?: number;
  updated_at?: number;
};

type RagCollection = {
  id?: string;
  name: string;
  qdrant_name: string;
  created_at?: number;
};

type ServiceSnapshot = {
  name?: string;
  status?: string;
  pid?: number;
  [key: string]: unknown;
};

function savedConfig(): Config {
  return {
    apiBase: localStorage.getItem("apiBase") || import.meta.env.VITE_API_BASE || window.location.origin,
    token: localStorage.getItem("token") || "",
    adminKey: localStorage.getItem("adminKey") || "",
  };
}

function newId(prefix: string) {
  return `${prefix}-${crypto.randomUUID?.() ?? crypto.getRandomValues(new Uint32Array(4)).join("-")}`;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorMessage(value: unknown): string {
  if (value instanceof Error) return value.message;
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    const nested = object.error && typeof object.error === "object" ? (object.error as Record<string, unknown>) : null;
    return String(nested?.message || object.detail || object.message || JSON.stringify(value));
  }
  return "Request failed";
}

function formatDate(ts?: number) {
  return ts ? new Date(ts * 1000).toLocaleString() : "";
}

function App() {
  const location = useLocation();
  const [config, setConfig] = useState<Config>(savedConfig);
  const [health, setHealth] = useState("offline");
  const [user, setUser] = useState<User | null>(null);
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);

  const apiBase = useMemo(() => config.apiBase.replace(/\/$/, ""), [config.apiBase]);
  const title = useMemo(() => titleFor(location.pathname), [location.pathname]);

  const notify = useCallback((message: string) => {
    setToast(message);
    window.clearTimeout((notify as unknown as { timer?: number }).timer);
    (notify as unknown as { timer?: number }).timer = window.setTimeout(() => setToast(""), 3600);
  }, []);

  const request = useCallback(
    async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
      const headers = new Headers(init.headers);
      if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
      if (config.token) headers.set("Authorization", `Bearer ${config.token}`);
      const response = await fetch(`${apiBase}${path}`, { ...init, headers });
      const text = await response.text();
      const data = text ? safeJson(text) : {};
      if (!response.ok) throw new Error(errorMessage(data));
      return data as T;
    },
    [apiBase, config.token],
  );

  const adminRequest = useCallback(
    async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
      const headers = new Headers(init.headers);
      headers.set("Content-Type", "application/json");
      headers.set("x-admin-key", config.adminKey);
      const response = await fetch(`${apiBase}${path}`, { ...init, headers });
      const text = await response.text();
      const data = text ? safeJson(text) : {};
      if (!response.ok) throw new Error(errorMessage(data));
      return data as T;
    },
    [apiBase, config.adminKey],
  );

  const refreshMe = useCallback(async () => {
    try {
      const data = await request<{ status?: string }>("/health");
      setHealth(data.status || "ok");
    } catch {
      setHealth("offline");
    }

    if (!config.token) {
      setUser(null);
      return;
    }

    try {
      setUser(await request<User>("/auth/me"));
    } catch (err) {
      setUser(null);
      notify(errorMessage(err));
    }
  }, [config.token, notify, request]);

  useEffect(() => {
    localStorage.setItem("apiBase", config.apiBase);
    localStorage.setItem("token", config.token);
    localStorage.setItem("adminKey", config.adminKey);
  }, [config]);

  useEffect(() => {
    void refreshMe();
  }, [refreshMe]);

  async function auth(mode: "login" | "register", email: string, password: string) {
    setBusy(true);
    try {
      const data = await request<{ access_token: string }>(`/auth/${mode}`, {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      setConfig((current) => ({ ...current, token: data.access_token }));
      notify("Signed in");
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    try {
      if (config.token) await request("/auth/logout", { method: "POST" });
    } catch {
      // Session may already be expired.
    }
    setConfig((current) => ({ ...current, token: "" }));
    setUser(null);
  }

  return (
    <div className="min-h-screen bg-[#f6f7f9]">
      <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[292px_minmax(0,1fr)]">
        <Sidebar config={config} health={health} onConfig={setConfig} />
        <main className="grid content-start gap-5 p-4 sm:p-6">
          <Topbar title={title} user={user} busy={busy} onRefresh={refreshMe} onLogout={logout} />
          <AuthPanel busy={busy} onAuth={(mode, email, password) => auth(mode, email, password).catch((err) => notify(errorMessage(err)))} />
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<ChatPage request={request} apiBase={apiBase} token={config.token} notify={notify} />} />
            <Route path="/video" element={<VideoPage request={request} apiBase={apiBase} token={config.token} notify={notify} />} />
            <Route path="/rag" element={<RagPage request={request} notify={notify} />} />
            <Route path="/history" element={<HistoryPage request={request} notify={notify} />} />
            <Route path="/admin" element={<AdminPage adminRequest={adminRequest} notify={notify} />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </main>
      </div>
      <Toast message={toast} />
    </div>
  );
}

function Sidebar({
  config,
  health,
  onConfig,
}: {
  config: Config;
  health: string;
  onConfig: (config: Config) => void;
}) {
  const location = useLocation();
  const [draft, setDraft] = useState(config);

  useEffect(() => setDraft(config), [config]);

  return (
    <aside className="bg-[#101820] p-4 text-slate-100 lg:min-h-screen">
      <div className="grid gap-5">
        <div className="flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-lg bg-ocean text-sm font-black">AI</div>
          <div className="min-w-0">
            <div className="font-semibold">Gateway</div>
            <div className="flex items-center gap-1.5 text-xs text-slate-300">
              <Activity size={13} />
              <span>{health}</span>
            </div>
          </div>
        </div>

        <nav className="grid gap-1" aria-label="Primary">
          <NavItem to="/chat" icon={<MessageSquare size={18} />} active={location.pathname.startsWith("/chat")}>
            Chat
          </NavItem>
          <NavItem to="/video" icon={<Video size={18} />} active={location.pathname.startsWith("/video")}>
            Video
          </NavItem>
          <NavItem to="/rag" icon={<Database size={18} />} active={location.pathname.startsWith("/rag")}>
            RAG
          </NavItem>
          <NavItem to="/history" icon={<History size={18} />} active={location.pathname.startsWith("/history")}>
            History
          </NavItem>
          <NavItem to="/admin" icon={<Shield size={18} />} active={location.pathname.startsWith("/admin")}>
            Admin
          </NavItem>
        </nav>

        <form
          className="grid gap-3 border-t border-white/10 pt-4"
          onSubmit={(event) => {
            event.preventDefault();
            onConfig({ ...draft, apiBase: draft.apiBase.replace(/\/$/, "") });
          }}
        >
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Settings size={16} />
            Runtime
          </div>
          <label className="field text-slate-300">
            API Base
            <input className="input border-white/[0.15] bg-white/10 text-white placeholder:text-slate-400" value={draft.apiBase} onChange={(event) => setDraft({ ...draft, apiBase: event.target.value })} />
          </label>
          <label className="field text-slate-300">
            Session
            <input className="input border-white/[0.15] bg-white/10 text-white placeholder:text-slate-400" value={draft.token} onChange={(event) => setDraft({ ...draft, token: event.target.value })} />
          </label>
          <label className="field text-slate-300">
            Admin Key
            <input className="input border-white/[0.15] bg-white/10 text-white placeholder:text-slate-400" value={draft.adminKey} onChange={(event) => setDraft({ ...draft, adminKey: event.target.value })} />
          </label>
          <button className="btn border-white/[0.15] bg-white/10 text-white hover:bg-white/[0.15] hover:text-white" type="submit">
            <CheckCircle2 size={16} />
            Save
          </button>
        </form>
      </div>
    </aside>
  );
}

function NavItem({ to, icon, active, children }: { to: string; icon: ReactNode; active: boolean; children: ReactNode }) {
  return (
    <Link
      className={`flex min-h-10 items-center gap-3 rounded-md px-3 text-sm font-semibold transition ${
        active ? "bg-white/[0.12] text-white" : "text-slate-300 hover:bg-white/[0.08] hover:text-white"
      }`}
      to={to}
    >
      {icon}
      <span>{children}</span>
    </Link>
  );
}

function Topbar({
  title,
  user,
  busy,
  onRefresh,
  onLogout,
}: {
  title: string;
  user: User | null;
  busy: boolean;
  onRefresh: () => void;
  onLogout: () => void;
}) {
  return (
    <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold text-ink">{title}</h1>
        <p className="mt-1 truncate text-sm text-muted">{user ? `${user.email} - ${user.user_id}` : "Not signed in"}</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="btn" type="button" onClick={onRefresh} disabled={busy}>
          <RefreshCw size={16} className={busy ? "animate-spin" : ""} />
          Reload
        </button>
        <button className="btn" type="button" onClick={onLogout}>
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </header>
  );
}

function AuthPanel({
  busy,
  onAuth,
}: {
  busy: boolean;
  onAuth: (mode: "login" | "register", email: string, password: string) => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <section className="surface">
      <form className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto_auto]" onSubmit={(event) => event.preventDefault()}>
        <label className="field">
          Email
          <input className="input" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} />
        </label>
        <label className="field">
          Password
          <input className="input" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} />
        </label>
        <button className="btn btn-primary self-end" type="button" disabled={busy} onClick={() => onAuth("login", email.trim(), password)}>
          <KeyRound size={16} />
          Login
        </button>
        <button className="btn self-end" type="button" disabled={busy} onClick={() => onAuth("register", email.trim(), password)}>
          Register
        </button>
      </form>
    </section>
  );
}

function ChatPage({
  request,
  apiBase,
  token,
  notify,
}: {
  request: <T>(path: string, init?: RequestInit) => Promise<T>;
  apiBase: string;
  token: string;
  notify: (message: string) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(defaultModel);
  const [sessionId, setSessionId] = useState(localStorage.getItem("sessionId") || newId("sess"));
  const [ragCollection, setRagCollection] = useState("");
  const [stream, setStream] = useState(true);
  const [useRag, setUseRag] = useState(false);
  const [sending, setSending] = useState(false);

  useEffect(() => localStorage.setItem("sessionId", sessionId), [sessionId]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const content = input.trim();
    if (!content) return;
    const assistantId = newId("msg");
    const payload = {
      model,
      session_id: sessionId,
      messages: [{ role: "user", content }],
      stream,
      use_rag: useRag,
      rag_collection: ragCollection || undefined,
      max_tokens: 1024,
      temperature: 0.7,
      top_p: 0.95,
    };
    setMessages((current) => [
      ...current,
      { id: newId("msg"), role: "user", content },
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setInput("");
    setSending(true);
    try {
      if (stream) {
        await streamChat(apiBase, token, payload, (chunk) => {
          setMessages((current) => current.map((message) => (message.id === assistantId ? { ...message, content: message.content + chunk } : message)));
        });
      } else {
        const data = await request<{ choices?: Array<{ message?: { content?: string } }> }>("/v1/chat/completions", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        const text = data.choices?.[0]?.message?.content || "";
        setMessages((current) => current.map((message) => (message.id === assistantId ? { ...message, content: text } : message)));
      }
    } catch (err) {
      notify(errorMessage(err));
      setMessages((current) => current.map((message) => (message.id === assistantId ? { ...message, content: errorMessage(err) } : message)));
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="grid gap-4">
      <div className="grid min-h-[620px] grid-rows-[1fr_auto] gap-4 rounded-lg border border-line bg-white p-4 shadow-panel">
        <div className="grid max-h-[56vh] min-h-80 content-start gap-3 overflow-auto rounded-lg border border-line bg-slate-50 p-3" aria-live="polite">
          {messages.length === 0 ? <EmptyState icon={<MessageSquare size={22} />} title="Chat" /> : null}
          {messages.map((message) => (
            <div key={message.id} className={`max-w-[78ch] rounded-lg border bg-white p-3 ${message.role === "user" ? "justify-self-end border-teal-200" : "justify-self-start border-blue-200"}`}>
              <div className="mb-1 text-xs font-bold text-muted">{message.role}</div>
              <div className="whitespace-pre-wrap text-sm leading-6">{message.content || (message.role === "assistant" ? "..." : "")}</div>
            </div>
          ))}
        </div>

        <form className="grid gap-3" onSubmit={submit}>
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)_auto_auto]">
            <label className="field">
              Model
              <input className="input" value={model} onChange={(event) => setModel(event.target.value)} />
            </label>
            <label className="field">
              Session
              <input className="input" value={sessionId} onChange={(event) => setSessionId(event.target.value)} />
            </label>
            <label className="field">
              RAG Collection
              <input className="input" value={ragCollection} onChange={(event) => setRagCollection(event.target.value)} />
            </label>
            <Toggle label="Stream" checked={stream} onChange={setStream} />
            <Toggle label="RAG" checked={useRag} onChange={setUseRag} />
          </div>
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
            <textarea className="textarea min-h-28" value={input} onChange={(event) => setInput(event.target.value)} />
            <button className="btn btn-primary self-end" type="submit" disabled={sending}>
              {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
              Send
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}

async function streamChat(apiBase: string, token: string, payload: Record<string, unknown>, onChunk: (chunk: string) => void) {
  const response = await fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) throw new Error(await response.text());

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      const json = safeJson(data) as { choices?: Array<{ delta?: { content?: string } }> };
      onChunk(json.choices?.[0]?.delta?.content || "");
    }
  }
}

function VideoPage({
  request,
  apiBase,
  token,
  notify,
}: {
  request: <T>(path: string, init?: RequestInit) => Promise<T>;
  apiBase: string;
  token: string;
  notify: (message: string) => void;
}) {
  const [form, setForm] = useState({
    mode: "text_to_video",
    width: "1024",
    height: "576",
    frames: "97",
    steps: "40",
    cfg: "7.5",
    imageUrl: "",
    videoUrl: "",
    audioUrl: "",
    keyframes: "",
    retakeStart: "",
    retakeEnd: "",
    prompt: "",
    negative: "",
  });
  const [jobs, setJobs] = useState<VideoJob[]>(() => {
    const stored = localStorage.getItem("videoJobs");
    return stored ? (JSON.parse(stored) as VideoJob[]) : [];
  });

  useEffect(() => localStorage.setItem("videoJobs", JSON.stringify(jobs.slice(0, 50))), [jobs]);

  function patch(key: keyof typeof form, value: string) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const payload: Record<string, unknown> = {
      mode: form.mode,
      prompt: form.prompt,
      negative_prompt: form.negative,
      width: Number(form.width),
      height: Number(form.height),
      num_frames: Number(form.frames),
      num_inference_steps: Number(form.steps),
      guidance_scale: Number(form.cfg),
    };
    if (form.imageUrl) payload.image_url = form.imageUrl;
    if (form.videoUrl) payload.video_url = form.videoUrl;
    if (form.audioUrl) payload.audio_url = form.audioUrl;
    const keyframes = form.keyframes.split(",").map((value) => value.trim()).filter(Boolean);
    if (keyframes.length) payload.keyframe_urls = keyframes;
    if (form.retakeStart) payload.retake_start_time = Number(form.retakeStart);
    if (form.retakeEnd) payload.retake_end_time = Number(form.retakeEnd);

    try {
      const queued = await request<{ job_id: string; status: string; created_at: number }>("/v1/video/jobs", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const job = { job_id: queued.job_id, status: queued.status, progress: 0, created_at: queued.created_at };
      setJobs((current) => [job, ...current.filter((item) => item.job_id !== job.job_id)]);
      void watchVideoJob(apiBase, token, job.job_id, (update) => setJobs((current) => current.map((item) => (item.job_id === update.job_id ? { ...item, ...update } : item))));
      notify(`Queued ${queued.job_id}`);
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  async function refreshJob(jobId: string) {
    try {
      const job = await request<VideoJob>(`/v1/video/jobs/${jobId}`);
      setJobs((current) => current.map((item) => (item.job_id === jobId ? { ...item, ...job } : item)));
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  async function cancelJob(jobId: string) {
    try {
      await request(`/v1/video/jobs/${jobId}`, { method: "DELETE" });
      await refreshJob(jobId);
      notify(`Cancel requested for ${jobId}`);
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  return (
    <section className="grid gap-4">
      <div className="surface">
        <form className="grid gap-3" onSubmit={submit}>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
            <label className="field">
              Mode
              <select className="input" value={form.mode} onChange={(event) => patch("mode", event.target.value)}>
                {videoModes.map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </label>
            <NumberField label="Width" value={form.width} step={32} onChange={(value) => patch("width", value)} />
            <NumberField label="Height" value={form.height} step={32} onChange={(value) => patch("height", value)} />
            <NumberField label="Frames" value={form.frames} step={8} onChange={(value) => patch("frames", value)} />
            <NumberField label="Steps" value={form.steps} step={1} onChange={(value) => patch("steps", value)} />
            <NumberField label="CFG" value={form.cfg} step={0.1} onChange={(value) => patch("cfg", value)} />
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            <TextField label="Image URL" value={form.imageUrl} onChange={(value) => patch("imageUrl", value)} />
            <TextField label="Video URL" value={form.videoUrl} onChange={(value) => patch("videoUrl", value)} />
            <TextField label="Audio URL" value={form.audioUrl} onChange={(value) => patch("audioUrl", value)} />
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            <TextField label="Keyframes" value={form.keyframes} onChange={(value) => patch("keyframes", value)} />
            <NumberField label="Retake Start" value={form.retakeStart} step={0.1} onChange={(value) => patch("retakeStart", value)} />
            <NumberField label="Retake End" value={form.retakeEnd} step={0.1} onChange={(value) => patch("retakeEnd", value)} />
          </div>
          <label className="field">
            Prompt
            <textarea className="textarea" value={form.prompt} onChange={(event) => patch("prompt", event.target.value)} />
          </label>
          <label className="field">
            Negative
            <textarea className="textarea min-h-16" value={form.negative} onChange={(event) => patch("negative", event.target.value)} />
          </label>
          <div>
            <button className="btn btn-primary" type="submit">
              <Play size={16} />
              Queue Video
            </button>
          </div>
        </form>
      </div>

      <div className="grid gap-3">
        <div className="flex items-center justify-between">
          <h2 className="section-title">Jobs</h2>
          <span className="muted">{jobs.length}</span>
        </div>
        {jobs.length === 0 ? <EmptyState icon={<Video size={22} />} title="No jobs" /> : null}
        <div className="grid gap-3">
          {jobs.map((job) => (
            <div key={job.job_id} className="rounded-lg border border-line bg-white p-4">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="min-w-0">
                  <div className="break-all text-sm font-semibold">{job.job_id}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
                    <Clock3 size={14} />
                    <span>{job.status}</span>
                    <span>{Math.round((job.progress || 0) * 100)}%</span>
                    <span>{formatDate(job.updated_at || job.created_at)}</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button className="btn" type="button" onClick={() => void refreshJob(job.job_id)}>
                    <RefreshCw size={16} />
                    Poll
                  </button>
                  <button className="btn btn-danger" type="button" onClick={() => void cancelJob(job.job_id)}>
                    <Square size={16} />
                    Cancel
                  </button>
                </div>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full bg-ocean" style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} />
              </div>
              {job.result_url ? (
                <a className="mt-3 block break-all text-sm font-semibold text-steel" href={job.result_url} target="_blank" rel="noreferrer">
                  {job.result_url}
                </a>
              ) : null}
              {job.error ? <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{job.error}</div> : null}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

async function watchVideoJob(apiBase: string, token: string, jobId: string, onUpdate: (job: VideoJob) => void) {
  const response = await fetch(`${apiBase}/v1/video/jobs/${jobId}/events`, {
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });
  if (!response.ok || !response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const job = safeJson(line.slice(6)) as VideoJob;
      if (job.job_id) onUpdate(job);
      if (["complete", "failed", "cancelled"].includes(job.status)) return;
    }
  }
}

function RagPage({ request, notify }: { request: <T>(path: string, init?: RequestInit) => Promise<T>; notify: (message: string) => void }) {
  const [name, setName] = useState("");
  const [ingestCollection, setIngestCollection] = useState("");
  const [source, setSource] = useState("api");
  const [text, setText] = useState("");
  const [collections, setCollections] = useState<RagCollection[]>([]);

  const load = useCallback(async () => {
    const data = await request<{ collections: RagCollection[] }>("/rag/collections");
    setCollections(data.collections || []);
  }, [request]);

  useEffect(() => {
    void load().catch((err) => notify(errorMessage(err)));
  }, [load, notify]);

  async function create(event: FormEvent) {
    event.preventDefault();
    try {
      await request("/rag/collections", { method: "POST", body: JSON.stringify({ name }) });
      setIngestCollection(name);
      setName("");
      await load();
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  async function ingest(event: FormEvent) {
    event.preventDefault();
    try {
      const data = await request<{ ingested_chunks?: number }>("/rag/ingest", {
        method: "POST",
        body: JSON.stringify({ collection: ingestCollection, source_name: source || "api", texts: [text], metadata: {} }),
      });
      notify(`Ingested ${data.ingested_chunks || 0} chunks`);
      setText("");
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  return (
    <section className="grid gap-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(280px,420px)_minmax(0,1fr)]">
        <form className="surface grid content-start gap-3" onSubmit={create}>
          <h2 className="section-title">Collection</h2>
          <TextField label="Name" value={name} onChange={setName} />
          <button className="btn btn-primary justify-self-start" type="submit">
            <Database size={16} />
            Create
          </button>
        </form>

        <form className="surface grid gap-3" onSubmit={ingest}>
          <h2 className="section-title">Ingest</h2>
          <div className="grid gap-3 md:grid-cols-2">
            <TextField label="Collection" value={ingestCollection} onChange={setIngestCollection} />
            <TextField label="Source" value={source} onChange={setSource} />
          </div>
          <label className="field">
            Text
            <textarea className="textarea min-h-44" value={text} onChange={(event) => setText(event.target.value)} />
          </label>
          <button className="btn btn-primary justify-self-start" type="submit">
            <Upload size={16} />
            Ingest
          </button>
        </form>
      </div>

      <div className="grid gap-3">
        <div className="flex items-center justify-between">
          <h2 className="section-title">Collections</h2>
          <button className="btn" type="button" onClick={() => void load().catch((err) => notify(errorMessage(err)))}>
            <RefreshCw size={16} />
            Reload
          </button>
        </div>
        {collections.length === 0 ? <EmptyState icon={<Database size={22} />} title="No collections" /> : null}
        <div className="grid gap-3">
          {collections.map((collection) => (
            <div key={collection.id || collection.qdrant_name} className="rounded-lg border border-line bg-white p-4">
              <div className="font-semibold">{collection.name}</div>
              <code className="mt-1 block break-all text-xs text-muted">{collection.qdrant_name}</code>
              <div className="mt-1 text-xs text-muted">{formatDate(collection.created_at)}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function HistoryPage({ request, notify }: { request: <T>(path: string, init?: RequestInit) => Promise<T>; notify: (message: string) => void }) {
  const [kind, setKind] = useState<"sessions" | "messages" | "videos">("sessions");
  const [sessionId, setSessionId] = useState("");
  const [data, setData] = useState<unknown>(null);

  async function load(nextKind = kind) {
    try {
      const path =
        nextKind === "sessions"
          ? "/history/sessions"
          : nextKind === "messages"
            ? `/history/chat${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`
            : "/history/videos";
      setKind(nextKind);
      setData(await request(path));
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  return (
    <section className="grid gap-4">
      <div className="surface">
        <div className="flex flex-wrap items-end gap-3">
          <button className={`btn ${kind === "sessions" ? "btn-primary" : ""}`} type="button" onClick={() => void load("sessions")}>
            <History size={16} />
            Sessions
          </button>
          <button className={`btn ${kind === "messages" ? "btn-primary" : ""}`} type="button" onClick={() => void load("messages")}>
            <MessageSquare size={16} />
            Messages
          </button>
          <button className={`btn ${kind === "videos" ? "btn-primary" : ""}`} type="button" onClick={() => void load("videos")}>
            <Video size={16} />
            Videos
          </button>
          <label className="field min-w-[260px]">
            Session
            <input className="input" value={sessionId} onChange={(event) => setSessionId(event.target.value)} />
          </label>
        </div>
      </div>
      <DataView value={data} />
    </section>
  );
}

function AdminPage({
  adminRequest,
  notify,
}: {
  adminRequest: <T>(path: string, init?: RequestInit) => Promise<T>;
  notify: (message: string) => void;
}) {
  const [model, setModel] = useState(defaultModel);
  const [output, setOutput] = useState<unknown>(null);

  async function run(path: string, init?: RequestInit) {
    try {
      setOutput(await adminRequest(path, init));
    } catch (err) {
      notify(errorMessage(err));
      setOutput({ error: errorMessage(err) });
    }
  }

  async function startModel(event: FormEvent) {
    event.preventDefault();
    await run(`/admin/models/${encodeURIComponent(model)}/start`, { method: "POST" });
  }

  return (
    <section className="grid gap-4">
      <div className="surface grid gap-4">
        <div className="flex flex-wrap gap-2">
          <button className="btn" type="button" onClick={() => void run("/status")}>
            <Activity size={16} />
            Status
          </button>
          <button className="btn" type="button" onClick={() => void run("/admin/gpus")}>
            <Server size={16} />
            GPUs
          </button>
          <button className="btn" type="button" onClick={() => void run("/admin/services")}>
            <Settings size={16} />
            Services
          </button>
          <button className="btn" type="button" onClick={() => void run("/admin/services/text-worker/start", { method: "POST" })}>
            <Play size={16} />
            Text
          </button>
          <button className="btn" type="button" onClick={() => void run("/admin/services/ltx-worker/start", { method: "POST" })}>
            <Video size={16} />
            LTX
          </button>
          <button className="btn btn-danger" type="button" onClick={() => void run("/admin/services/ltx-worker/stop", { method: "POST" })}>
            <Square size={16} />
            Stop LTX
          </button>
        </div>
        <form className="grid gap-3 md:grid-cols-[minmax(0,520px)_auto]" onSubmit={startModel}>
          <TextField label="Model" value={model} onChange={setModel} />
          <button className="btn btn-primary self-end" type="submit">
            <RotateCw size={16} />
            Start Model
          </button>
        </form>
      </div>
      <DataView value={output} />
    </section>
  );
}

function DataView({ value }: { value: unknown }) {
  if (value == null) return <EmptyState icon={<Activity size={22} />} title="No data" />;
  const arrays = flattenArrays(value);
  if (arrays.length > 0) {
    return (
      <div className="grid gap-3">
        {arrays.map((item, index) => (
          <div key={index} className="rounded-lg border border-line bg-white p-4">
            <pre className="max-h-72 overflow-auto text-xs leading-5 text-ink">{JSON.stringify(item, null, 2)}</pre>
          </div>
        ))}
      </div>
    );
  }
  return <pre className="code-block">{JSON.stringify(value, null, 2)}</pre>;
}

function flattenArrays(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== "object") return [];
  return Object.values(value as Record<string, unknown>).flatMap((item) => (Array.isArray(item) ? item : []));
}

function TextField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field">
      {label}
      <input className="input" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function NumberField({ label, value, step, onChange }: { label: string; value: string; step: number; onChange: (value: string) => void }) {
  return (
    <label className="field">
      {label}
      <input className="input" type="number" step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex min-h-10 items-center gap-2 self-end rounded-md border border-line bg-white px-3 text-sm font-semibold text-ink">
      <input className="h-4 w-4 accent-ocean" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function EmptyState({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="grid min-h-28 place-items-center rounded-lg border border-dashed border-line bg-white p-6 text-center text-muted">
      <div className="grid gap-2 justify-items-center">
        {icon}
        <span className="text-sm font-semibold">{title}</span>
      </div>
    </div>
  );
}

function Toast({ message }: { message: string }) {
  return (
    <div
      className={`fixed bottom-4 right-4 z-50 max-w-md rounded-lg bg-[#101820] px-4 py-3 text-sm font-semibold text-white shadow-panel transition ${
        message ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0"
      }`}
      role="status"
      aria-live="polite"
    >
      {message}
    </div>
  );
}

function titleFor(path: string) {
  if (path.startsWith("/video")) return "Video";
  if (path.startsWith("/rag")) return "RAG";
  if (path.startsWith("/history")) return "History";
  if (path.startsWith("/admin")) return "Admin";
  return "Chat";
}

export default App;
