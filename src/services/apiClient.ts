/**
 * Cliente de Integración API - Sistema de Inventario Inteligente Tiendas 3A
 * Conecta el MVP (React/Vite en Vercel) con el Backend de Machine Learning (Flask en Render)
 * y la Base de Datos Cloud (Supabase PostgreSQL).
 * Cumple con el Contrato de Integración de la Rúbrica APF3 (Sección 6.3).
 */

export interface MLPredictionRequest {
  sku: string;
  producto_id: string;
  precio: number;
  costo: number;
  stock_actual: number;
  capacidad_gondola?: number;
  pack_size: number;
  categoria_id: string;
  lead_time_dias: number;
  lag_1d: number;
  lag_7d: number;
  media_movil_7d?: number;
  std_movil_7d?: number;
  dia_semana?: number;
  es_fin_de_semana?: number;
  es_quincena?: number;
  nivel_servicio?: number;
}

export interface MLPredictionResponse {
  sku: string;
  producto_id: string;
  demanda_predicha_dia: number;
  demanda_predicha_semana: number;
  stock_seguridad_ss: number;
  punto_reorden_rop: number;
  stock_actual: number;
  dias_cobertura: number;
  riesgo: 'critico' | 'alto' | 'medio' | 'bajo';
  estado_alerta: string;
  requiere_orden: boolean;
  pedido_sugerido_bultos: number;
  pedido_sugerido_unidades: number;
  costo_pedido_estimado: number;
  modelo: string;
  mape_modelo: number;
  timestamp: string;
}

export interface BackendHealth {
  status: string;
  service: string;
  model_loaded: boolean;
  model_algorithm: string;
  model_version: string;
  database_connected: boolean;
  timestamp: string;
}

function resolveApiBaseUrl(): string {
  const envUrl = (import.meta.env.VITE_API_URL || '').trim();
  if (envUrl) {
    let clean = envUrl.replace(/\/+$/, '');
    // Si la web corre en HTTPS (Vercel) y la URL del backend empieza por http://, forzar https:// para evitar que el navegador bloquee por Mixed Content
    if (typeof window !== 'undefined' && window.location.protocol === 'https:' && clean.startsWith('http://')) {
      clean = clean.replace(/^http:\/\//, 'https://');
    }
    return clean;
  }
  return 'http://localhost:5000';
}

export const API_BASE_URL = resolveApiBaseUrl();

/** Verifica disponibilidad del backend Flask / Render (permite hasta 15s si Render está en cold start) */
export async function checkBackendHealth(timeoutMs = 12000): Promise<BackendHealth | null> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!res.ok) {
      console.warn(`[ML API Health] Respuesta no exitosa: HTTP ${res.status}`);
      return null;
    }
    return await res.json();
  } catch (err: any) {
    console.warn(`[ML API Health] No se pudo contactar a ${API_BASE_URL}/health:`, err?.message || err);
    return null;
  }
}

/** Ejecuta inferencia en vivo con el modelo serializado en Python (soporta hasta 50s por cold start de Render) */
export async function predictWithML(
  payload: MLPredictionRequest,
  onSlowWakeup?: (isWaking: boolean) => void
): Promise<MLPredictionResponse | null> {
  try {
    const controller = new AbortController();
    // Avisar si Render tarda más de 2 segundos en responder (indicativo de cold start en Render Free)
    const slowWakeupTimer = setTimeout(() => {
      onSlowWakeup?.(true);
    }, 2500);

    const timeoutId = setTimeout(() => controller.abort(), 50000);
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(slowWakeupTimer);
    clearTimeout(timeoutId);
    onSlowWakeup?.(false);
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    onSlowWakeup?.(false);
    console.warn('[ML API] Fallback a motor local TS:', err);
    return null;
  }
}

/** Guarda un movimiento en Supabase mediante la API */
export async function persistMovement(movement: {
  id: string;
  ts: number;
  productId: string;
  type: string;
  qty: number;
  user: string;
  reason?: string;
  amount?: number;
  gondolaAfter: number;
  almacenAfter: number;
}): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/movements`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(movement),
    });
    return res.ok;
  } catch {
    return false;
  }
}
