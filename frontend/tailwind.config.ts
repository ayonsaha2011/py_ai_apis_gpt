import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17202a",
        muted: "#607086",
        line: "#d9e0e8",
        ocean: "#0d766e",
        steel: "#235a97",
      },
      boxShadow: {
        panel: "0 12px 30px rgba(22, 32, 42, 0.08)",
      },
    },
  },
  plugins: [],
} satisfies Config;
