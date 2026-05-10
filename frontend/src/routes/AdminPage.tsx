import { Activity, FileText, Play, RefreshCw, RotateCw, Server, Settings, Shield, Square, Video } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { errorMessage } from "../api";
import { DataView, EmptyState, NumberField, TextField } from "../components/ui";
import type { AdminUser } from "../types";

const defaultModel = "google/gemma-3-12b-it-qat-q4_0-unquantized";

type AdminPageProps = {
  adminRequest: <T>(path: string, init?: RequestInit) => Promise<T>;
  notify: (message: string) => void;
};

export function AdminPage({ adminRequest, notify }: AdminPageProps) {
  const [model, setModel] = useState(defaultModel);
  const [output, setOutput] = useState<unknown>(null);
  const [logService, setLogService] = useState("ltx");
  const [logLines, setLogLines] = useState("200");
  const [logs, setLogs] = useState<{ service?: string; path?: string; content?: string } | null>(null);
  const [tab, setTab] = useState<"services" | "logs" | "users" | "models">("services");
  const [users, setUsers] = useState<AdminUser[]>([]);

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

  async function loadLogs() {
    try {
      const data = await adminRequest<{ service?: string; path?: string; content?: string }>(`/admin/logs/${logService}?lines=${encodeURIComponent(logLines)}`);
      setLogs(data);
    } catch (err) {
      notify(errorMessage(err));
      setLogs({ content: errorMessage(err) });
    }
  }

  async function loadUsers() {
    try {
      const data = await adminRequest<{ users?: AdminUser[] }>("/admin/users");
      setUsers(data.users || []);
      setOutput(data);
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  async function setUserRole(userId: string, role: "admin" | "user") {
    try {
      await adminRequest(`/admin/users/${encodeURIComponent(userId)}/role`, {
        method: "PATCH",
        body: JSON.stringify({ role }),
      });
      await loadUsers();
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  useEffect(() => {
    if (tab === "users" && users.length === 0) void loadUsers();
  }, [tab]);

  return (
    <section className="grid gap-4">
      <div className="surface flex flex-wrap gap-2">
        {(["services", "logs", "users", "models"] as const).map((item) => (
          <button key={item} className={`btn ${tab === item ? "btn-primary" : ""}`} type="button" onClick={() => setTab(item)}>
            {item === "services" ? <Settings size={16} /> : item === "logs" ? <FileText size={16} /> : item === "users" ? <Shield size={16} /> : <RotateCw size={16} />}
            {item[0].toUpperCase() + item.slice(1)}
          </button>
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_460px]">
        <div className={`surface grid gap-4 ${tab === "services" ? "" : "hidden"}`}>
          <div className="grid gap-2 sm:grid-cols-3">
            <button className="btn min-h-12 justify-start" type="button" onClick={() => void run("/status")}>
              <Activity size={16} />
              Status
            </button>
            <button className="btn min-h-12 justify-start" type="button" onClick={() => void run("/admin/gpus")}>
              <Server size={16} />
              GPUs
            </button>
            <button className="btn min-h-12 justify-start" type="button" onClick={() => void run("/admin/services")}>
              <Settings size={16} />
              Services
            </button>
            <button className="btn min-h-12 justify-start" type="button" onClick={() => void run("/admin/services/text-worker/start", { method: "POST" })}>
              <Play size={16} />
              Start text
            </button>
            <button className="btn min-h-12 justify-start" type="button" onClick={() => void run("/admin/services/ltx-worker/start", { method: "POST" })}>
              <Video size={16} />
              Start LTX
            </button>
            <button className="btn btn-danger min-h-12 justify-start" type="button" onClick={() => void run("/admin/services/ltx-worker/stop", { method: "POST" })}>
              <Square size={16} />
              Stop LTX
            </button>
          </div>
        </div>

        <div className={`surface grid gap-4 ${tab === "models" ? "" : "hidden"}`}>
          <form className="grid gap-3 md:grid-cols-[minmax(0,520px)_auto]" onSubmit={startModel}>
            <TextField label="Model" value={model} onChange={setModel} />
            <button className="btn btn-primary self-end" type="submit">
              <RotateCw size={16} />
              Start Model
            </button>
          </form>
        </div>

        <div className={`surface grid content-start gap-3 ${tab === "logs" ? "" : "hidden"}`}>
          <div className="flex items-center gap-2">
            <FileText size={18} />
            <h2 className="section-title">Server logs</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-[1fr_110px]">
            <label className="field">
              Service
              <select className="input" value={logService} onChange={(event) => setLogService(event.target.value)}>
                <option value="gateway">gateway</option>
                <option value="text">text</option>
                <option value="ltx">ltx</option>
                <option value="qdrant">qdrant</option>
              </select>
            </label>
            <NumberField label="Lines" value={logLines} step={50} onChange={setLogLines} />
          </div>
          <button className="btn btn-primary justify-self-start" type="button" onClick={() => void loadLogs()}>
            <RefreshCw size={16} />
            Load logs
          </button>
          {logs ? (
            <div className="grid gap-2">
              <div className="truncate text-xs font-semibold text-muted">{logs.path || logs.service}</div>
              <pre className="code-block max-h-[520px] whitespace-pre-wrap">{logs.content || ""}</pre>
            </div>
          ) : null}
        </div>

        <div className={`surface grid content-start gap-3 xl:col-span-2 ${tab === "users" ? "" : "hidden"}`}>
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Shield size={18} />
              <h2 className="section-title">Users</h2>
            </div>
            <button className="btn" type="button" onClick={() => void loadUsers()}>
              <RefreshCw size={16} />
              Reload
            </button>
          </div>
          {users.length === 0 ? <EmptyState icon={<Shield size={22} />} title="No users loaded" /> : null}
          <div className="grid gap-2">
            {users.map((item) => (
              <div key={item.user_id} className="grid gap-3 rounded-lg border border-line bg-white p-3 md:grid-cols-[minmax(0,1fr)_150px] md:items-center">
                <div className="min-w-0">
                  <div className="truncate font-semibold text-ink">{item.email}</div>
                  <div className="mt-1 break-all text-xs text-muted">{item.user_id}</div>
                  <div className="mt-1 text-xs text-muted">{formatDate(item.created_at)}</div>
                </div>
                <select className="input" value={item.role} onChange={(event) => void setUserRole(item.user_id, event.target.value as "admin" | "user")}>
                  <option value="user">user</option>
                  <option value="admin">admin</option>
                </select>
              </div>
            ))}
          </div>
        </div>
      </div>
      <DataView value={output} />
    </section>
  );
}

function formatDate(ts?: number) {
  return ts ? new Date(ts * 1000).toLocaleString() : "";
}
