export type Config = {
  apiBase: string;
  token: string;
  adminKey: string;
};

export type User = {
  user_id: string;
  email: string;
  role: "admin" | "user" | string;
  created_at: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type VideoJob = {
  job_id: string;
  status: string;
  progress?: number;
  result_url?: string | null;
  error?: string | null;
  created_at?: number;
  updated_at?: number;
  effective_seed?: number;
  r2_key?: string;
  metadata?: {
    upscaled?: boolean;
    render_width?: number;
    render_height?: number;
    output_width?: number;
    output_height?: number;
    [key: string]: unknown;
  };
};

export type RagCollection = {
  id?: string;
  name: string;
  qdrant_name: string;
  created_at?: number;
};

export type ServiceSnapshot = {
  name?: string;
  status?: string;
  pid?: number;
  [key: string]: unknown;
};

export type AdminUser = {
  user_id: string;
  email: string;
  role: "admin" | "user" | string;
  created_at: number;
};
