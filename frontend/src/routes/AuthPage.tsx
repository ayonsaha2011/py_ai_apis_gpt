import { KeyRound, Loader2, UserPlus } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { errorMessage } from "../api";

type AuthPageProps = {
  mode: "login" | "register";
  busy: boolean;
  notify: (message: string) => void;
  onAuth: (email: string, password: string) => Promise<void>;
};

export function AuthPage({ mode, busy, notify, onAuth }: AuthPageProps) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const isRegister = mode === "register";

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (password.length < 12) {
      notify("Password must be at least 12 characters.");
      return;
    }
    if (isRegister && password !== confirmPassword) {
      notify("Passwords do not match.");
      return;
    }
    try {
      await onAuth(email.trim(), password);
      navigate("/chat");
    } catch (err) {
      notify(errorMessage(err));
    }
  }

  return (
    <section className="grid min-h-[62vh] place-items-center">
      <form className="surface grid w-full max-w-[520px] gap-5" onSubmit={submit}>
        <div>
          <div className="grid h-12 w-12 place-items-center rounded-lg bg-ocean text-white">
            {isRegister ? <UserPlus size={22} /> : <KeyRound size={22} />}
          </div>
          <h2 className="mt-4 text-2xl font-semibold text-ink">{isRegister ? "Create account" : "Login"}</h2>
          <p className="mt-1 text-sm text-muted">{isRegister ? "Create a gateway account for chat, RAG, video history, and artifacts." : "Use your gateway account to continue."}</p>
        </div>

        <div className="grid gap-3">
          <label className="field">
            Email
            <input className="input" autoComplete="email" required type="email" value={email} onChange={(event) => setEmail(event.target.value)} />
          </label>
          <label className="field">
            Password
            <input className="input" autoComplete={isRegister ? "new-password" : "current-password"} required minLength={12} type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          {isRegister ? (
            <label className="field">
              Confirm password
              <input className="input" autoComplete="new-password" required minLength={12} type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} />
            </label>
          ) : null}
        </div>

        <button className="btn btn-primary w-full" type="submit" disabled={busy}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : isRegister ? <UserPlus size={16} /> : <KeyRound size={16} />}
          {isRegister ? "Create account" : "Login"}
        </button>

        <div className="text-center text-sm text-muted">
          {isRegister ? "Already have an account?" : "Need an account?"}{" "}
          <Link className="font-semibold text-ocean" to={isRegister ? "/login" : "/register"}>
            {isRegister ? "Login" : "Register"}
          </Link>
        </div>
      </form>
    </section>
  );
}
