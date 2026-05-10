import { Activity, CheckCircle2, Clock3 } from "lucide-react";
import { ReactNode } from "react";

export function QualityPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`inline-flex min-h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs font-bold ${ok ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"}`}>
      {ok ? <CheckCircle2 size={14} /> : <Clock3 size={14} />}
      {label}
    </span>
  );
}

export function SetupRow({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-line bg-slate-50 px-3 py-2">
      <div className="flex items-center gap-2 text-muted">
        {icon}
        <span>{label}</span>
      </div>
      <span className="truncate font-semibold text-ink">{value}</span>
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  const style = normalized === "complete"
    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
    : normalized === "failed"
      ? "border-red-200 bg-red-50 text-red-700"
      : normalized === "cancelled"
        ? "border-slate-200 bg-slate-100 text-slate-600"
        : "border-blue-200 bg-blue-50 text-blue-700";
  return <span className={`rounded-md border px-2 py-0.5 font-bold ${style}`}>{status}</span>;
}

export function DataView({ value }: { value: unknown }) {
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

export function TextField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field">
      {label}
      <input className="input" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

export function NumberField({ label, value, step, onChange }: { label: string; value: string; step: number; onChange: (value: string) => void }) {
  return (
    <label className="field">
      {label}
      <input className="input" type="number" step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex min-h-10 items-center gap-2 self-end rounded-md border border-line bg-white px-3 text-sm font-semibold text-ink">
      <input className="h-4 w-4 accent-ocean" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

export function EmptyState({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="grid min-h-28 place-items-center rounded-lg border border-dashed border-line bg-white p-6 text-center text-muted">
      <div className="grid gap-2 justify-items-center">
        {icon}
        <span className="text-sm font-semibold">{title}</span>
      </div>
    </div>
  );
}

export function Toast({ message }: { message: string }) {
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
