import {
  Activity,
  Clapperboard,
  CheckCircle2,
  Clock3,
  Database,
  History,
  Image as ImageIcon,
  KeyRound,
  Loader2,
  LogOut,
  MessageSquare,
  Music2,
  Play,
  RefreshCw,
  Send,
  Server,
  Settings,
  Shield,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
  UserPlus,
  Video,
  Wand2,
} from "lucide-react";
import { FormEvent, ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { errorMessage, safeJson } from "./api";
import { DataView, EmptyState, NumberField, QualityPill, SetupRow, StatusBadge, TextField, Toast, Toggle } from "./components/ui";
import { AdminPage } from "./routes/AdminPage";
import { AuthPage } from "./routes/AuthPage";
import type { ChatMessage, Config, RagCollection, User, VideoJob } from "./types";

const defaultModel = "google/gemma-3-12b-it-qat-q4_0-unquantized";
const videoModes = [
  "text_to_video",
  "image_to_video",
] as const;

type VideoMode = (typeof videoModes)[number];

const modeLabels: Record<VideoMode, string> = {
  text_to_video: "Text to video",
  image_to_video: "Image to video",
};

const resolutionPresets = [
  { label: "SD wide", width: "768", height: "448" },
  { label: "HD", width: "1024", height: "576" },
  { label: "H200 20s", width: "1408", height: "768" },
  { label: "B200 Full-HD 20s", width: "1920", height: "1088" },
  { label: "Full-HD 5s", width: "1920", height: "1088" },
  { label: "Square", width: "768", height: "768" },
  { label: "Portrait", width: "576", height: "1024" },
];

const durationOptions = [
  { label: "5 seconds", value: "5", frames: "121" },
  { label: "10 seconds", value: "10", frames: "241" },
  { label: "15 seconds", value: "15", frames: "361" },
  { label: "20 seconds", value: "20", frames: "481" },
] as const;

type VideoExample = {
  title: string;
  mode: VideoMode;
  prompt: string;
  negative: string;
  width: string;
  height: string;
  duration: string;
  frames: string;
  steps: string;
  cfg: string;
};

type VideoForm = {
  mode: VideoMode;
  width: string;
  height: string;
  duration: string;
  frames: string;
  steps: string;
  cfg: string;
  seedHint: string;
  enhancePrompt: boolean;
  imageUrl: string;
  videoUrl: string;
  audioUrl: string;
  keyframes: string;
  retakeStart: string;
  retakeEnd: string;
  prompt: string;
  negative: string;
};

const videoExamples: VideoExample[] = [
  {
    title: "Product macro",
    mode: "text_to_video",
    width: "1024",
    height: "576",
    duration: "5",
    frames: "121",
    steps: "40",
    cfg: "7.5",
    prompt:
      "A cinematic macro product shot of a matte black wireless headphone rotating slowly on polished graphite, soft studio reflections, shallow depth of field, controlled dolly-in camera movement, premium commercial lighting.",
    negative: "low quality, blurry, warped geometry, unreadable text, flicker, jitter, noisy shadows",
  },
  {
    title: "Character moment",
    mode: "text_to_video",
    width: "768",
    height: "1024",
    duration: "5",
    frames: "121",
    steps: "40",
    cfg: "7.0",
    prompt:
      "A young astronaut standing inside a greenhouse on Mars, orange dust drifting outside the glass, plants moving gently from the ventilation, subtle handheld camera, realistic suit fabric, warm practical lights.",
    negative: "cartoon, plastic skin, extra fingers, distorted helmet, fast camera, harsh flicker",
  },
  {
    title: "City motion",
    mode: "text_to_video",
    width: "768",
    height: "448",
    duration: "5",
    frames: "121",
    steps: "36",
    cfg: "7.0",
    prompt:
      "A late-night aerial shot flying between rain-slick skyscrapers in Tokyo, neon signs reflected on glass, traffic trails below, smooth forward motion, realistic atmosphere, cinematic contrast.",
    negative: "overexposed neon, unstable camera, smeared buildings, low detail, text artifacts",
  },
  {
    title: "Nature detail",
    mode: "text_to_video",
    width: "768",
    height: "448",
    duration: "5",
    frames: "121",
    steps: "36",
    cfg: "7.0",
    prompt:
      "A close shot of dew drops sliding across a green leaf after sunrise, tiny insects moving in the background, soft golden light, macro lens, slow lateral camera movement, realistic bokeh.",
    negative: "blur, oversaturated colors, artificial leaf texture, frame jumps, noisy background",
  },
  {
    title: "HD showroom",
    mode: "text_to_video",
    width: "1024",
    height: "576",
    duration: "5",
    frames: "121",
    steps: "36",
    cfg: "7.0",
    prompt:
      "A cinematic HD walk-through of a modern electric vehicle showroom at night, camera gliding from the glass entrance toward a silver concept car, reflections moving across polished floors, soft ceiling panels, subtle people silhouettes in the background, realistic commercial lighting, slow continuous motion.",
    negative: "low quality, warped car body, flicker, jitter, smeared reflections, unreadable signage, duplicated wheels, fast cuts",
  },
  {
    title: "HD food story",
    mode: "text_to_video",
    width: "1024",
    height: "576",
    duration: "5",
    frames: "121",
    steps: "36",
    cfg: "7.0",
    prompt:
      "An HD restaurant kitchen sequence, close camera drifting past fresh herbs, a chef plating handmade pasta, steam rising under warm lamps, sauce poured slowly, final hero plate landing on a dark stone counter, natural hand motion, shallow depth of field, premium documentary style.",
    negative: "messy hands, distorted utensils, flickering steam, overexposed highlights, warped plates, low detail food texture",
  },
  {
    title: "HD travel reveal",
    mode: "text_to_video",
    width: "1024",
    height: "576",
    duration: "5",
    frames: "121",
    steps: "36",
    cfg: "7.0",
    prompt:
      "An HD travel reveal at sunrise, camera starts behind linen curtains inside a quiet coastal hotel room, moves toward an open balcony, ocean waves and palm trees appear, sunlight spreads across the floor, curtains moving gently in the breeze, calm luxury resort atmosphere, smooth dolly movement.",
    negative: "cartoon look, harsh camera shake, warped balcony rails, noisy water, flicker, oversaturated sky, low quality fabric",
  },
  {
    title: "B200 Full-HD 20s",
    mode: "text_to_video",
    width: "1920",
    height: "1088",
    duration: "20",
    frames: "481",
    steps: "40",
    cfg: "7.5",
    prompt:
      "A full-HD 20-second continuous cinematic shot inside a futuristic robotics lab, camera slowly tracking from a workbench of precision tools toward a humanoid robot waking under soft blue-white lights, engineers moving naturally in the background, tiny status LEDs pulsing, realistic metal and glass materials, no cuts, smooth controlled dolly movement.",
    negative: "low quality, flicker, jitter, warped hands, duplicated robot parts, text artifacts, fast cuts, unstable camera",
  },
  {
    title: "H200 20s scene",
    mode: "text_to_video",
    width: "1408",
    height: "768",
    duration: "20",
    frames: "481",
    steps: "40",
    cfg: "7.5",
    prompt:
      "A long continuous shot following a lone hiker through a misty mountain forest at dawn, sunlight breaking through pine trees, light fog rolling across the trail, slow steady tracking camera behind the figure, realistic footstep motion, natural ambient light shifting as clouds move, cinematic depth.",
    negative: "camera shake, fast cuts, warped trees, overexposed sky, ghosting, motion blur artifacts, cartoon",
  },
  {
    title: "Full-HD 5s reveal",
    mode: "text_to_video",
    width: "1920",
    height: "1088",
    duration: "5",
    frames: "121",
    steps: "40",
    cfg: "7.5",
    prompt:
      "A cinematic full-HD opening shot of a luxury penthouse living room at golden hour, camera slowly pushing in from the floor-to-ceiling window toward a glass coffee table, city skyline behind, warm diffused light on polished concrete floors, minimal furniture, no people, ultra-realistic materials.",
    negative: "low detail, flicker, warped reflections, cartoon furniture, oversaturated sky, fast camera, jitter",
  },
];

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

function formatDate(ts?: number) {
  return ts ? new Date(ts * 1000).toLocaleString() : "";
}

function framesForDuration(duration: string) {
  return durationOptions.find((option) => option.value === duration)?.frames || "121";
}


function _tokenCount(width: number, height: number, frames: number): number {
  return (Math.floor((frames - 1) / 8) + 1) * Math.floor(height / 8) * Math.floor(width / 8);
}

function _maxTokens(profile: string): number {
  const p = profile.toLowerCase();
  if (p.includes("b200")) return 2_100_000;
  if (p.includes("h200")) return 1_770_000;
  if (p.includes("h100")) return 540_000;
  return 140_000;
}

function _suggestNative(frames: number, cap: number): [number, number] {
  const fFactor = Math.floor((frames - 1) / 8) + 1;
  for (let w = 3840; w >= 128; w -= 32) {
    const h = Math.round((w * 9) / 16 / 32) * 32;
    if (h < 128) continue;
    if (fFactor * Math.floor(h / 8) * Math.floor(w / 8) <= cap) return [w, h];
  }
  return [256, 144];
}

function durationBlockedForProfile(_option: (typeof durationOptions)[number], _profile: string): boolean {
  return false;
}

function durationOptionLabel(option: (typeof durationOptions)[number], _profile: string): string {
  return option.label;
}

function videoBudgetWarning(form: VideoForm, profile: string): string {
  const width = Number(form.width);
  const height = Number(form.height);
  const frames = Number(form.frames);
  if (!Number.isFinite(width) || !Number.isFinite(height) || !Number.isFinite(frames)) return "";
  const tokens = _tokenCount(width, height, frames);
  const cap = _maxTokens(profile);
  if (tokens > cap) {
    const [sw, sh] = _suggestNative(frames, cap);
    return `${width}x${height}@${frames}f exceeds ${profile || "GPU"} capacity (${tokens.toLocaleString()} > ${cap.toLocaleString()} tokens). Suggested: ${sw}x${sh}@${frames}f`;
  }
  return "";
}

function videoGenerationMode(form: VideoForm, profile: string): string {
  return videoBudgetWarning(form, profile) ? "Blocked" : "Native";
}


function videoSizeValid(form: VideoForm, _profile: string): boolean {
  const width = Number(form.width);
  const height = Number(form.height);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return false;
  return width % 32 === 0 && height % 32 === 0;
}

function defaultVideoForm(profile: string): VideoForm {
  const b200 = profile.toLowerCase().includes("b200");
  return {
    mode: "text_to_video",
    width: b200 ? "1920" : "1024",
    height: b200 ? "1088" : "576",
    duration: b200 ? "20" : "5",
    frames: b200 ? "481" : "121",
    steps: "40",
    cfg: "7.5",
    seedHint: "",
    enhancePrompt: false,
    imageUrl: "",
    videoUrl: "",
    audioUrl: "",
    keyframes: "",
    retakeStart: "",
    retakeEnd: "",
    prompt: "",
    negative: "",
  };
}

function App() {
  const location = useLocation();
  const [config, setConfig] = useState<Config>(savedConfig);
  const [health, setHealth] = useState("offline");
  const [gatewayProfile, setGatewayProfile] = useState("cloud_h200");
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);

  const apiBase = useMemo(() => config.apiBase.replace(/\/$/, ""), [config.apiBase]);
  const isAdmin = user?.role === "admin";
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
      if (config.token) headers.set("Authorization", `Bearer ${config.token}`);
      if (config.adminKey) headers.set("x-admin-key", config.adminKey);
      const response = await fetch(`${apiBase}${path}`, { ...init, headers });
      const text = await response.text();
      const data = text ? safeJson(text) : {};
      if (!response.ok) throw new Error(errorMessage(data));
      return data as T;
    },
    [apiBase, config.adminKey, config.token],
  );

  const refreshMe = useCallback(async () => {
    try {
      const data = await request<{ status?: string; profile?: string }>("/status");
      setHealth(data.status || "ok");
      if (data.profile) setGatewayProfile(data.profile);
    } catch {
      setHealth("offline");
    }

    if (!config.token) {
      setUser(null);
      setAuthReady(true);
      return;
    }

    try {
      setUser(await request<User>("/auth/me"));
    } catch (err) {
      setUser(null);
      notify(errorMessage(err));
    } finally {
      setAuthReady(true);
    }
  }, [config.token, notify, request]);

  useEffect(() => {
    localStorage.setItem("apiBase", config.apiBase);
    localStorage.setItem("token", config.token);
    localStorage.setItem("adminKey", config.adminKey);
  }, [config]);

  useEffect(() => {
    setAuthReady(false);
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
      setAuthReady(false);
      notify(mode === "register" ? "Account created" : "Signed in");
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
    setAuthReady(true);
  }

  return (
    <div className="min-h-screen bg-[#f6f7f9]">
      <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[292px_minmax(0,1fr)]">
        <Sidebar config={config} health={health} user={user} onConfig={setConfig} />
        <main className="grid content-start gap-5 p-4 sm:p-6">
          <Topbar title={title} user={user} busy={busy} onRefresh={refreshMe} onLogout={logout} />
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/login" element={<AuthPage mode="login" busy={busy} notify={notify} onAuth={(email, password) => auth("login", email, password)} />} />
            <Route path="/register" element={<AuthPage mode="register" busy={busy} notify={notify} onAuth={(email, password) => auth("register", email, password)} />} />
            <Route path="/chat" element={<ChatPage request={request} apiBase={apiBase} token={config.token} notify={notify} />} />
            <Route path="/video" element={<VideoPage request={request} apiBase={apiBase} token={config.token} profile={gatewayProfile} notify={notify} />} />
            <Route path="/rag" element={<RagPage request={request} notify={notify} />} />
            <Route path="/history" element={<HistoryPage request={request} notify={notify} />} />
            <Route
              path="/admin"
              element={
                !authReady && config.token ? (
                  <EmptyState icon={<Shield size={22} />} title="Checking access" />
                ) : isAdmin ? (
                  <AdminPage adminRequest={adminRequest} notify={notify} />
                ) : (
                  <Navigate to="/chat" replace />
                )
              }
            />
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
  user,
  onConfig,
}: {
  config: Config;
  health: string;
  user: User | null;
  onConfig: (config: Config) => void;
}) {
  const location = useLocation();
  const [draft, setDraft] = useState(config);
  const isAdmin = user?.role === "admin";

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
          <NavItem to="/login" icon={<KeyRound size={18} />} active={location.pathname.startsWith("/login")}>
            Login
          </NavItem>
          <NavItem to="/register" icon={<UserPlus size={18} />} active={location.pathname.startsWith("/register")}>
            Register
          </NavItem>
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
          {isAdmin ? (
            <NavItem to="/admin" icon={<Shield size={18} />} active={location.pathname.startsWith("/admin")}>
              Admin
            </NavItem>
          ) : null}
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
          {isAdmin ? (
            <label className="field text-slate-300">
              Session
              <input className="input border-white/[0.15] bg-white/10 text-white placeholder:text-slate-400" value={draft.token} onChange={(event) => setDraft({ ...draft, token: event.target.value })} />
            </label>
          ) : null}
          {isAdmin ? (
            <label className="field text-slate-300">
              Admin Key
              <input className="input border-white/[0.15] bg-white/10 text-white placeholder:text-slate-400" value={draft.adminKey} onChange={(event) => setDraft({ ...draft, adminKey: event.target.value })} />
            </label>
          ) : null}
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
        <p className="mt-1 truncate text-sm text-muted">{user ? `${user.email} - ${user.role}` : "Not signed in"}</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="btn" type="button" onClick={onRefresh} disabled={busy}>
          <RefreshCw size={16} className={busy ? "animate-spin" : ""} />
          Reload
        </button>
        {user ? (
          <button className="btn" type="button" onClick={onLogout}>
            <LogOut size={16} />
            Logout
          </button>
        ) : (
          <>
            <Link className="btn" to="/login">
              <KeyRound size={16} />
              Login
            </Link>
            <Link className="btn btn-primary" to="/register">
              <UserPlus size={16} />
              Register
            </Link>
          </>
        )}
      </div>
    </header>
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
  profile,
  notify,
}: {
  request: <T>(path: string, init?: RequestInit) => Promise<T>;
  apiBase: string;
  token: string;
  profile: string;
  notify: (message: string) => void;
}) {
  const [form, setForm] = useState<VideoForm>(() => defaultVideoForm(profile));
  const [jobs, setJobs] = useState<VideoJob[]>(() => {
    const stored = localStorage.getItem("videoJobs");
    return stored ? (JSON.parse(stored) as VideoJob[]) : [];
  });

  useEffect(() => localStorage.setItem("videoJobs", JSON.stringify(jobs.slice(0, 50))), [jobs]);

  const mediaRequirements = videoRequirements(form.mode);
  const frameValid = /^\d+$/.test(form.frames) && (Number(form.frames) - 1) % 8 === 0;
  const sizeValid = videoSizeValid(form, profile);
  const budgetWarning = videoBudgetWarning(form, profile);
  const generationMode = videoGenerationMode(form, profile);

  useEffect(() => {
    const option = durationOptions.find((item) => item.value === form.duration);
    if (option && durationBlockedForProfile(option, profile)) {
      setForm((current) => ({ ...current, duration: "5", frames: "121" }));
    }
  }, [form.duration, profile]);

  useEffect(() => {
    if (!profile.toLowerCase().includes("b200")) return;
    if (form.prompt || form.negative || form.seedHint) return;
    if (form.width === "1024" && form.height === "576" && form.duration === "5" && form.frames === "121") {
      setForm(defaultVideoForm(profile));
    }
  }, [form.duration, form.frames, form.height, form.negative, form.prompt, form.seedHint, form.width, profile]);

  function patch<K extends keyof VideoForm>(key: K, value: VideoForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function applyPreset(preset: (typeof resolutionPresets)[number]) {
    setForm((current) => ({ ...current, width: preset.width, height: preset.height }));
  }

  function setDuration(duration: string) {
    setForm((current) => ({ ...current, duration, frames: framesForDuration(duration) }));
  }

  function applyExample(example: VideoExample) {
    setForm((current) => ({
      ...current,
      mode: example.mode,
      width: example.width,
      height: example.height,
      duration: example.duration,
      frames: example.frames,
      steps: example.steps,
      cfg: example.cfg,
      prompt: example.prompt,
      negative: example.negative,
    }));
    notify(`Loaded ${example.title}`);
  }

  function resetForm() {
    setForm(defaultVideoForm(profile));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!sizeValid || !frameValid) {
      notify("Video size must be valid for native or upscaled output, and frames must satisfy 8k+1.");
      return;
    }
    if (budgetWarning) {
      notify(budgetWarning);
      return;
    }
    const payload: Record<string, unknown> = {
      mode: form.mode,
      prompt: form.prompt,
      negative_prompt: form.negative,
      width: Number(form.width),
      height: Number(form.height),
      num_frames: Number(form.frames),
      num_inference_steps: Number(form.steps),
      guidance_scale: Number(form.cfg),
      enhance_prompt: form.enhancePrompt,
    };
    if (form.seedHint) payload.seed_hint = Number(form.seedHint);
    if (form.imageUrl) payload.image_url = form.imageUrl;
    if (form.videoUrl) payload.video_url = form.videoUrl;
    if (form.audioUrl) payload.audio_url = form.audioUrl;
    const keyframes = form.keyframes.split(",").map((value) => value.trim()).filter(Boolean);
    if (keyframes.length) payload.keyframe_urls = keyframes;
    if (form.retakeStart) payload.retake_start_time = Number(form.retakeStart);
    if (form.retakeEnd) payload.retake_end_time = Number(form.retakeEnd);

    try {
      const queued = await request<{ job_id: string; status: string; created_at: number; effective_seed?: number; r2_key?: string }>("/v1/video/jobs", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const job = {
        job_id: queued.job_id,
        status: queued.status,
        progress: 0,
        created_at: queued.created_at,
        effective_seed: queued.effective_seed,
        r2_key: queued.r2_key,
      };
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
    <section className="grid gap-5">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="surface grid gap-5">
          <div className="flex flex-col gap-3 border-b border-line pb-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="flex items-center gap-2 text-lg font-semibold text-ink">
                <Clapperboard size={20} />
                Video job
              </div>
              <p className="mt-1 text-sm text-muted">Unique queued generations with gateway validation and SSE progress.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <QualityPill ok={sizeValid} label="size valid" />
              <QualityPill ok={frameValid} label="8k+1 frames" />
              <QualityPill ok={!budgetWarning} label="GPU budget" />
            </div>
          </div>

          <form className="grid gap-5" onSubmit={submit}>
            <div className="grid gap-4">
              <div className="grid gap-2">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-semibold text-muted">Mode</span>
                  <span className="text-sm font-semibold text-ocean">{modeLabels[form.mode]}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  {videoModes.map((mode) => (
                    <button
                      className={`min-h-11 rounded-md border px-3 text-sm font-semibold transition ${
                        form.mode === mode ? "border-ocean bg-ocean text-white shadow-panel" : "border-line bg-white text-ink hover:border-ocean hover:text-ocean"
                      }`}
                      key={mode}
                      type="button"
                      onClick={() => patch("mode", mode)}
                    >
                      {modeLabels[mode]}
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {resolutionPresets.map((preset) => (
                  <button className="btn min-h-10 px-2 text-xs" key={preset.label} type="button" onClick={() => applyPreset(preset)}>
                    {preset.label}
                  </button>
                ))}
              </div>
              <div className="rounded-md border border-line bg-slate-50 px-3 py-2 text-sm font-semibold text-muted">
                Output mode: <span className="text-ink">{generationMode}</span>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
              <NumberField label="Width" value={form.width} step={32} onChange={(value) => patch("width", value)} />
              <NumberField label="Height" value={form.height} step={32} onChange={(value) => patch("height", value)} />
              <label className="field">
                Duration
                <select className="input" value={form.duration} onChange={(event) => setDuration(event.target.value)}>
                  {durationOptions.map((option) => (
                    <option key={option.value} value={option.value} disabled={durationBlockedForProfile(option, profile)}>
                      {durationOptionLabel(option, profile)}
                    </option>
                  ))}
                </select>
              </label>
              <NumberField label="Steps" value={form.steps} step={1} onChange={(value) => patch("steps", value)} />
              <NumberField label="CFG" value={form.cfg} step={0.1} onChange={(value) => patch("cfg", value)} />
              <NumberField label="Seed hint" value={form.seedHint} step={1} onChange={(value) => patch("seedHint", value)} />
            </div>

            <div className="rounded-lg border border-line bg-slate-50 p-3">
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-ink">
                <Upload size={16} />
                Inputs for {modeLabels[form.mode]}
              </div>
              <div className="grid gap-3 lg:grid-cols-3">
                {mediaRequirements.image ? <TextField label="Image URL" value={form.imageUrl} onChange={(value) => patch("imageUrl", value)} /> : null}
                {mediaRequirements.video ? <TextField label="Video URL" value={form.videoUrl} onChange={(value) => patch("videoUrl", value)} /> : null}
                {mediaRequirements.audio ? <TextField label="Audio URL" value={form.audioUrl} onChange={(value) => patch("audioUrl", value)} /> : null}
                {mediaRequirements.keyframes ? <TextField label="Keyframe URLs" value={form.keyframes} onChange={(value) => patch("keyframes", value)} /> : null}
                {mediaRequirements.retake ? <NumberField label="Retake start" value={form.retakeStart} step={0.1} onChange={(value) => patch("retakeStart", value)} /> : null}
                {mediaRequirements.retake ? <NumberField label="Retake end" value={form.retakeEnd} step={0.1} onChange={(value) => patch("retakeEnd", value)} /> : null}
                {!Object.values(mediaRequirements).some(Boolean) ? <div className="rounded-md border border-dashed border-line bg-white p-3 text-sm text-muted">No media URL required for this mode.</div> : null}
              </div>
            </div>

            <label className="field">
              Prompt
              <textarea className="textarea min-h-36 text-base leading-7" value={form.prompt} onChange={(event) => patch("prompt", event.target.value)} />
            </label>
            {budgetWarning ? (
              <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm font-semibold leading-6 text-amber-800">
                {budgetWarning}
              </div>
            ) : null}
            <label className="field">
              Negative prompt
              <textarea className="textarea min-h-20" value={form.negative} onChange={(event) => patch("negative", event.target.value)} />
            </label>

            <div className="flex flex-col gap-3 border-t border-line pt-4 sm:flex-row sm:items-center sm:justify-between">
              <Toggle label="Enhance prompt" checked={form.enhancePrompt} onChange={(checked) => patch("enhancePrompt", checked)} />
              <div className="flex flex-wrap gap-2">
                <button className="btn" type="button" onClick={resetForm}>
                  <Trash2 size={16} />
                  Reset
                </button>
                <button className="btn btn-primary" type="submit" disabled={!sizeValid || !frameValid || Boolean(budgetWarning) || !form.prompt.trim()}>
                  <Play size={16} />
                  Queue video
                </button>
              </div>
            </div>
          </form>
        </div>

        <aside className="grid content-start gap-4">
          <div className="surface grid gap-3">
            <div className="flex items-center gap-2">
              <Sparkles size={18} />
              <h2 className="section-title">Example prompts</h2>
            </div>
            <div className="grid gap-2">
              {videoExamples.map((example) => (
                <button className="text-left" key={example.title} type="button" onClick={() => applyExample(example)}>
                  <div className="rounded-lg border border-line bg-white p-3 transition hover:border-ocean hover:shadow-panel">
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-semibold text-ink">{example.title}</div>
                      <Wand2 size={16} className="text-ocean" />
                    </div>
                    <div className="mt-1 text-xs font-semibold text-muted">{modeLabels[example.mode]} - {example.width}x{example.height} - {example.duration}s</div>
                    <p className="mt-2 line-clamp-3 text-sm leading-5 text-muted">{example.prompt}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="surface grid gap-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal size={18} />
              <h2 className="section-title">Current setup</h2>
            </div>
            <div className="grid gap-2 text-sm">
              <SetupRow icon={<Video size={15} />} label="Mode" value={modeLabels[form.mode]} />
              <SetupRow icon={<Server size={15} />} label="Profile" value={profile} />
              <SetupRow icon={<Sparkles size={15} />} label="Output" value={generationMode} />
              <SetupRow icon={<ImageIcon size={15} />} label="Canvas" value={`${form.width}x${form.height}`} />
              <SetupRow icon={<Clock3 size={15} />} label="Duration" value={`${form.duration}s`} />
              <SetupRow icon={<Clapperboard size={15} />} label="Frames" value={form.frames} />
              <SetupRow icon={<Music2 size={15} />} label="Media" value={inputSummary(mediaRequirements)} />
            </div>
          </div>
        </aside>
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
                    <StatusBadge status={job.status} />
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
              <div className="mt-3 grid gap-2 lg:grid-cols-[minmax(0,1fr)_auto]">
                <div className="flex flex-wrap gap-1.5">
                  {videoStatusTimeline.map((status) => (
                    <span key={status} className={`rounded-md border px-2 py-1 text-[11px] font-bold ${timelineActive(job.status, status) ? "border-ocean bg-ocean text-white" : "border-line bg-slate-50 text-muted"}`}>
                      {status}
                    </span>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <button className="btn px-2 py-1 text-xs" type="button" onClick={() => void copyText(job.job_id, notify)}>
                    Copy job
                  </button>
                  {job.r2_key ? (
                    <button className="btn px-2 py-1 text-xs" type="button" onClick={() => void copyText(job.r2_key || "", notify)}>
                      Copy R2
                    </button>
                  ) : null}
                </div>
              </div>
              <div className="mt-3 grid gap-2 text-xs text-muted sm:grid-cols-3">
                {job.effective_seed != null ? <SetupRow icon={<Sparkles size={15} />} label="Seed" value={String(job.effective_seed)} /> : null}
                {job.metadata?.render_width && job.metadata?.render_height ? <SetupRow icon={<Clapperboard size={15} />} label="Render" value={`${job.metadata.render_width}x${job.metadata.render_height}`} /> : null}
                {job.metadata?.output_width && job.metadata?.output_height ? <SetupRow icon={<ImageIcon size={15} />} label="Output" value={`${job.metadata.output_width}x${job.metadata.output_height}${job.metadata.upscaled ? " upscaled" : ""}`} /> : null}
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

function videoRequirements(mode: VideoMode) {
  return {
    image: mode === "image_to_video",
    video: false,
    audio: false,
    keyframes: false,
    retake: false,
  };
}

function inputSummary(requirements: ReturnType<typeof videoRequirements>) {
  const items = [
    requirements.image ? "image" : "",
    requirements.video ? "video" : "",
    requirements.audio ? "audio" : "",
    requirements.keyframes ? "keyframes" : "",
    requirements.retake ? "time range" : "",
  ].filter(Boolean);
  return items.length ? items.join(", ") : "prompt only";
}

const videoStatusTimeline = ["queued", "materializing_inputs", "generating", "encoding", "upscaling", "uploading", "complete"];

function timelineActive(current: string, step: string) {
  const currentIndex = videoStatusTimeline.indexOf(current);
  const stepIndex = videoStatusTimeline.indexOf(step);
  return current === step || (currentIndex >= 0 && stepIndex >= 0 && stepIndex <= currentIndex);
}

async function copyText(value: string, notify: (message: string) => void) {
  try {
    await navigator.clipboard.writeText(value);
    notify("Copied");
  } catch {
    notify(value);
  }
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

function titleFor(path: string) {
  if (path.startsWith("/login")) return "Login";
  if (path.startsWith("/register")) return "Register";
  if (path.startsWith("/video")) return "Video";
  if (path.startsWith("/rag")) return "RAG";
  if (path.startsWith("/history")) return "History";
  if (path.startsWith("/admin")) return "Admin";
  return "Chat";
}

export default App;
