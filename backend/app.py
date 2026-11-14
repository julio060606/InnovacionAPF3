"""
=============================================================================
API REST - SISTEMA DE INVENTARIO INTELIGENTE TIENDAS 3A (TANTAMAYO)
Backend en Flask + Inferencia Scikit-Learn (modelo_demanda_3a.joblib)
Conexión a Base de Datos Supabase PostgreSQL
Despliegue compatible con Render y Vercel
=============================================================================
"""

import os
import sys
import json
import math
import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor

# Asegurar UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = Flask(__name__)
# Permitir peticiones desde cualquier origen (Vercel, localhost, etc.)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_demanda_3a.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metricas_modelos.json")
ERRORS_PATH = os.path.join(BASE_DIR, "analisis_errores.json")

# Conexión a Supabase (Usa el Pooler IPv4 para compatibilidad total con Render Free Tier)
DEFAULT_DB_URI = "postgresql://postgres.uqkkjfsuvjwyyixcubmt:shTEgJDJSqDGpqG5@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
DB_URI = os.getenv("DATABASE_URL", DEFAULT_DB_URI)
# Si Render tiene configurada la URL antigua directa db.* que falla por IPv6, redirigir al pooler IPv4
if "db.uqkkjfsuvjwyyixcubmt.supabase.co" in DB_URI:
    DB_URI = DEFAULT_DB_URI

# Cargar el modelo serializado de Scikit-Learn
print(f"[BOOT] Cargando modelo serializado desde: {MODEL_PATH}")
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("✓ Modelo Gradient Boosting cargado exitosamente.")
except Exception as e:
    model_pipeline = None
    print(f"⚠ Advertencia: No se pudo cargar el modelo joblib: {e}")

def get_db_connection():
    """Retorna una conexión a Supabase PostgreSQL."""
    try:
        return psycopg2.connect(DB_URI, cursor_factory=RealDictCursor)
    except Exception as e:
        print(f"Error conectando a BD: {e}")
        return None

# =============================================================================
# ENDPOINTS REST (CONTRATO DE INTEGRACIÓN - SECCIÓN 6.3)
# =============================================================================

@app.route("/", methods=["GET", "OPTIONS"])
@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    """Healthcheck del servicio y verificación del modelo y base de datos."""
    db_ok = False
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                db_ok = True
        except Exception:
            pass
        finally:
            conn.close()

    return jsonify({
        "status": "online",
        "service": "API Inventario Inteligente Tiendas 3A (Sede Tantamayo)",
        "model_loaded": model_pipeline is not None,
        "model_algorithm": "GradientBoostingRegressor (scikit-learn)",
        "model_version": "1.0-APF3",
        "database_connected": db_ok,
        "timestamp": datetime.datetime.now().isoformat()
    }), 200

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    """
    CONTRATO DE INTEGRACIÓN (Sección 6.3):
    Recibe un registro de producto desde el MVP y devuelve la inferencia del modelo
    junto con los cálculos logísticos de ROP, Stock de Seguridad y Alertas.
    """
    if model_pipeline is None:
        return jsonify({"error": "El modelo no se encuentra cargado en el servidor."}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Cuerpo de la petición JSON vacío o inválido."}), 400

    try:
        # Extraer variables con valores por defecto coherentes
        precio = float(data.get("precio", 4.5))
        costo = float(data.get("costo", 3.1))
        lag_1d = float(data.get("lag_1d", 18.0))
        lag_7d = float(data.get("lag_7d", 20.0))
        media_movil_7d = float(data.get("media_movil_7d", (lag_1d + lag_7d) / 2))
        std_movil_7d = float(data.get("std_movil_7d", max(1.5, media_movil_7d * 0.15)))
        dia_semana = int(data.get("dia_semana", datetime.datetime.now().weekday()))
        es_fin_de_semana = int(data.get("es_fin_de_semana", 1 if dia_semana in (4, 5, 6) else 0))
        es_quincena = int(data.get("es_quincena", 1 if datetime.datetime.now().day in (15, 16, 30, 31, 1) else 0))
        categoria_id = str(data.get("categoria_id", "fv"))

        # Parámetros logísticos de Tiendas 3A
        lead_time_dias = int(data.get("lead_time_dias", 2))
        stock_actual = float(data.get("stock_actual", 15.0))
        pack_size = float(data.get("pack_size", 10.0))
        nivel_servicio = float(data.get("nivel_servicio", 0.95))

        # Crear DataFrame para el pipeline de Scikit-Learn
        df_input = pd.DataFrame([{
            "precio": precio,
            "costo": costo,
            "lag_1d": lag_1d,
            "lag_7d": lag_7d,
            "media_movil_7d": media_movil_7d,
            "std_movil_7d": std_movil_7d,
            "dia_semana": dia_semana,
            "es_fin_de_semana": es_fin_de_semana,
            "es_quincena": es_quincena,
            "categoria_id": categoria_id
        }])

        # Inferencia del modelo
        pred_demanda = float(model_pipeline.predict(df_input)[0])
        pred_demanda = max(0.1, round(pred_demanda, 2))

        # Cálculos de Operaciones Logísticas
        # z de distribución normal según nivel de servicio (90%=1.28, 95%=1.645, 99%=2.326)
        z = 1.645 if nivel_servicio >= 0.95 else 1.28
        # SS = z * sigma * sqrt(L)
        ss = round(z * std_movil_7d * math.sqrt(lead_time_dias), 2)
        # ROP = Demanda(L + 1) + SS
        lead_time_demand = round(pred_demanda * (lead_time_dias + 1), 2)
        rop = round(lead_time_demand + ss, 2)

        # Días de cobertura
        dias_cobertura = round(stock_actual / pred_demanda, 1) if pred_demanda > 0 else 99.0

        # Reglas de Alerta y Decisión
        requiere_orden = stock_actual <= rop
        alerta_quiebre = stock_actual <= ss

        if stock_actual <= 0:
            riesgo = "critico"
            estado_alerta = "QUIEBRE_ACTUAL"
        elif alerta_quiebre:
            riesgo = "alto"
            estado_alerta = "QUIEBRE_INMINENTE"
        elif requiere_orden:
            riesgo = "medio"
            estado_alerta = "EMITIR_ORDEN"
        else:
            riesgo = "bajo"
            estado_alerta = "OPTIMO"

        # Sugerencia de compra a bulto cerrado
        if requiere_orden:
            meta_stock = rop + (pred_demanda * 7) # Cobertura de 7 días
            unidades_necesarias = max(0.0, meta_stock - stock_actual)
            bultos_sugeridos = math.ceil(unidades_necesarias / pack_size)
            unidades_sugeridas = bultos_sugeridos * pack_size
        else:
            bultos_sugeridos = 0
            unidades_sugeridas = 0.0

        # Respuesta en formato JSON del Contrato de Integración
        return jsonify({
            "sku": data.get("sku", "SKU-3A"),
            "producto_id": data.get("producto_id", "fv-mandarina"),
            "demanda_predicha_dia": pred_demanda,
            "demanda_predicha_semana": round(pred_demanda * 7, 2),
            "stock_seguridad_ss": ss,
            "punto_reorden_rop": rop,
            "stock_actual": stock_actual,
            "dias_cobertura": dias_cobertura,
            "riesgo": riesgo,
            "estado_alerta": estado_alerta,
            "requiere_orden": requiere_orden,
            "pedido_sugerido_bultos": bultos_sugeridos,
            "pedido_sugerido_unidades": unidades_sugeridas,
            "costo_pedido_estimado": round(unidades_sugeridas * costo, 2),
            "modelo": "GradientBoostingRegressor",
            "mape_modelo": 15.63,
            "timestamp": datetime.datetime.now().isoformat()
        }), 200

    except Exception as ex:
        print(f"Error procesando predicción: {ex}")
        return jsonify({"error": f"Error interno en inferencia: {str(ex)}"}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Devuelve las métricas oficiales de entrenamiento y comparación."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    return jsonify({"error": "No se encontró el archivo de métricas."}), 404

@app.route("/api/errors", methods=["GET"])
def get_errors():
    """Devuelve el análisis de los 3 casos de fallo (Sección 6.1)."""
    if os.path.exists(ERRORS_PATH):
        with open(ERRORS_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    return jsonify({"error": "No se encontró el archivo de análisis de errores."}), 404

@app.route("/api/products", methods=["GET"])
def get_products():
    """Consulta los 33 productos desde Supabase."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "No se pudo conectar a la base de datos."}), 503
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.id, p.sku, p.codigo_barras, p.nombre, p.categoria_id, c.nombre as categoria_nombre,
                       p.unidad, p.precio, p.costo, p.stock_gondola, p.stock_almacen, p.capacidad_gondola,
                       p.pack_size, p.pack_label, p.lead_time_dias, p.ubicacion, p.abc, p.emoji
                FROM productos p
                JOIN categorias c ON p.categoria_id = c.id
                ORDER BY p.categoria_id, p.sku;
            """)
            rows = cur.fetchall()
            return jsonify(rows), 200
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    finally:
        conn.close()

@app.route("/api/movements", methods=["GET", "POST"])
def movements():
    """Permite listar y registrar movimientos del Kardex en Supabase (Persistencia)."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Base de datos no disponible."}), 503

    if request.method == "GET":
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT m.*, p.nombre as producto_nombre, p.sku
                    FROM movimientos_kardex m
                    JOIN productos p ON m.producto_id = p.id
                    ORDER BY m.ts DESC LIMIT 100;
                """)
                return jsonify(cur.fetchall()), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    elif request.method == "POST":
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Cuerpo inválido."}), 400

        try:
            with conn.cursor() as cur:
                m_id = data.get("id", f"MV-{int(datetime.datetime.now().timestamp()*1000)}")
                ts = int(data.get("ts", int(datetime.datetime.now().timestamp()*1000)))
                prod_id = data["productId"]
                tipo = data["type"]
                qty = float(data["qty"])
                user = data.get("user", "Colaborador 3A")
                reason = data.get("reason", "Operación manual")
                amount = float(data.get("amount", 0.0)) if data.get("amount") else None
                g_after = float(data.get("gondolaAfter", 0))
                a_after = float(data.get("almacenAfter", 0))

                cur.execute("""
                    INSERT INTO movimientos_kardex (
                        id, ts, producto_id, tipo, cantidad, usuario, motivo, monto,
                        gondola_despues, almacen_despues, origen
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'real')
                """, (m_id, ts, prod_id, tipo, qty, user, reason, amount, g_after, a_after))

                # Actualizar stock en la tabla productos
                cur.execute("""
                    UPDATE productos
                    SET stock_gondola = %s, stock_almacen = %s, updated_at = NOW()
                    WHERE id = %s
                """, (g_after, a_after, prod_id))

                conn.commit()
                return jsonify({"status": "created", "id": m_id}), 201
        except Exception as ex:
            conn.rollback()
            return jsonify({"error": str(ex)}), 500
        finally:
            conn.close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" Servidor Flask Tiendas 3A ejecutándose en puerto {port}")
    print(f" Endpoint Inferencia ML: POST http://localhost:{port}/api/predict")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
