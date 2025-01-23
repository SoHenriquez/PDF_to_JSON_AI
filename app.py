import os
import time
import json
import logging
from datetime import datetime
from google.colab import userdata
import google.generativeai as genai

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de Gemini
try:
    genai.configure(api_key=userdata.get('GEMINI_API_KEY'))
    logging.info("API Key de Gemini configurada correctamente.")
except Exception as e:
    logging.error(f"Error al configurar API Key de Gemini: {e}")
    raise e

def upload_to_gemini(path, mime_type=None, timeout=30):
    """
    Sube el archivo PDF a Gemini con timeout y reintentos.

    Args:
        path: Ruta al archivo
        mime_type: Tipo MIME del archivo
        timeout: Tiempo máximo de espera en segundos
    """
    start_time = time.time()
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Verificar si el archivo existe
            if not os.path.exists(path):
                raise FileNotFoundError(f"No se encontró el archivo: {path}")

            # Verificar el tamaño del archivo
            file_size = os.path.getsize(path)
            if file_size > 20 * 1024 * 1024:  # 20MB límite
                raise ValueError(f"El archivo es demasiado grande: {file_size/1024/1024:.2f}MB")

            logging.info(f"Intentando subir archivo {path} (Intento {retry_count + 1}/{max_retries})")

            # Intentar subir con timeout
            file = genai.upload_file(path, mime_type=mime_type)

            if file and hasattr(file, 'display_name'):
                logging.info(f"Archivo subido exitosamente: '{file.display_name}' como: {file.uri}")
                return file
            else:
                raise ValueError("La subida del archivo no devolvió un objeto válido")

        except Exception as e:
            retry_count += 1
            elapsed_time = time.time() - start_time

            if elapsed_time > timeout:
                logging.error(f"Timeout alcanzado después de {elapsed_time:.2f} segundos")
                raise TimeoutError(f"La subida del archivo excedió el tiempo límite de {timeout} segundos")

            if retry_count >= max_retries:
                logging.error(f"Error después de {max_retries} intentos: {str(e)}")
                raise

            # Esperar antes de reintentar
            wait_time = min(2 ** retry_count, 10)  # Backoff exponencial
            logging.warning(f"Error en intento {retry_count}: {str(e)}. Reintentando en {wait_time} segundos...")
            time.sleep(wait_time)

    return None

def wait_for_files_active(files):
    """Espera a que los archivos estén activos en Gemini."""
    logging.info("Esperando el procesamiento de los archivos...")
    for file in files:
        if not file:
            continue
        try:
            current_file = genai.get_file(file.name)
            while current_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                current_file = genai.get_file(file.name)
            if current_file.state.name != "ACTIVE":
                logging.error(f"Error al procesar archivo {current_file.name}: {current_file.state.name}")
                return False
            logging.info(f"Archivo '{current_file.name}' listo.")
        except Exception as e:
            logging.error(f"Error al chequear estado del archivo '{file.name}': {e}")
            return False
    print("\nTodos los archivos están listos")
    return True

# Configuración del modelo Gemini
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 20,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
    system_instruction=(
        "Eres un experto en extraer información de informes quirúrgicos. "
        "Tu tarea es analizar el documento y extraer los datos en el formato exacto solicitado, "
        "manteniendo la estructura y las etiquetas proporcionadas. NO DEJES INFORMACIÓN AFUERA."
    )
)

def procesar_informe_quirurgico(pdf_path):
    """Procesa un informe quirúrgico y extrae la información relevante."""
    try:
        files = [upload_to_gemini(pdf_path, mime_type="application/pdf")]

        if not wait_for_files_active(files):
            logging.error("Error en el procesamiento del archivo PDF.")
            return None

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [files[0]],
                }
            ]
        )

        # Prompt con la estructura deseada
        prompt = """
        Analiza el informe quirúrgico y extrae TODA la información en el siguiente formato exacto.
        Es CRUCIAL que captures TODOS los detalles, especialmente en las secciones extensas como
        Hallazgos_operatorios y Detalle_quirúrgico. NO omitas ninguna información:

        DATOS_BASICOS
        Ingreso: [valor]
        Número_Folio: [valor]
        Ubicación: [valor]

        IDENTIFICACION
        Tipo_documento: [valor]
        Número: [valor]
        Apellidos: [valor]
        Nombres: [valor]
        Edad: [valor]
        Dirección: [valor]
        Sexo_biológico: [valor]
        Género: [valor]
        Teléfono: [valor]
        Tipo_paciente: [valor]
        Tipo_afiliado: [valor]
        Profesión: [valor]
        Estado_civil: [valor]
        Entidad_responsable: [valor]
        Seguridad_social: [valor]
        Grupo_étnico: [valor]

        EQUIPO_QUIRURGICO
        Cirujano: [valor]
        Tipo_identificación_cirujano: [valor]
        Identificación_cirujano: [valor]
        Anestesiólogo: [valor]
        Tipo_identificación_anestesiólogo: [valor]
        Identificación_anestesiólogo: [valor]
        Instrumentador: [valor]
        Tipo_identificación_instrumentador: [valor]
        Identificación_instrumentador: [valor]
        Circulante: [valor]
        Tipo_identificación_circulante: [valor]
        Identificación_circulante: [valor]

        DIAGNOSTICOS
        Diagnósticos_Preoperatorios: [valor]
        Código_CIE10_Pre: [valor]
        Diagnósticos_Posoperatorios: [valor]
        Código_CIE10_Pos: [valor]

        INTERVENCION
        Fecha_hora_inicio: [valor]
        Fecha_hora_fin: [valor]
        Cirugía_urgente: [valor]
        Sala_cirugía: [valor]
        Tiempo_quirúrgico: [valor]
        Clasificación_ASA: [valor]
        Tipo_herida: [valor]
        Tipo_anestesia: [valor]
        Profilaxis_antimicrobianos: [valor]

        PROCEDIMIENTOS
        Código_CUPS: [valor]
        Descripción_procedimiento: [valor]
        Vía_abordaje: [valor]

        DESCRIPCION_QUIRURGICA
        Hallazgos_operatorios: [valor]
        Detalle_quirúrgico: [valor]
        Complicaciones: [valor]
        Materiales: [valor]
        Sangrado_perioperatorio: [valor]

        CONTEO_MATERIAL
        Compresas: [valor]
        Gasas: [valor]
        Tejidos_anatomía_patológica: [valor]

        DATOS_PROFESIONAL
        Nombre_profesional: [valor]
        Especialidad: [valor]
        Tipo_identificación: [valor]
        Identificación: [valor]
        Tarjeta_profesional: [valor]

        Reemplaza [valor] con la información correspondiente encontrada en el documento.
        Mantén exactamente el mismo formato, etiquetas y estructura.
        LA DESCRIPCIÓN DE CADA PARTE DEBE SER EXACTA.
        HALLAZGOS OPERATORIOS SUELE SER EXTENSA.
        EL DOCUMENTO TRAE TODO SU KEY: VALOR, POR LO QUE ES CASI IMPOSIBLE QUE ALGUNA KEY NO TENGA SU VALOR.
        IMPORTANTE:
        - Captura TODOS los hallazgos operatorios, sin omitir ningún detalle
        - Incluye el detalle quirúrgico completo, con todos los pasos del procedimiento
        - Mantén los saltos de línea originales usando '\n'
        - No resumas ni acortes ninguna sección
        - Si hay múltiples procedimientos o diagnósticos, inclúyelos todos
        - Preserva la estructura exacta del documento original
        """

        response = chat_session.send_message(prompt)
        return response.text.strip()

    except Exception as e:
        logging.error(f"Error al procesar el informe quirúrgico: {e}")
        return None

def guardar_en_txt(datos, output_file):
    """Guarda los datos extraídos en un archivo TXT."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for dato in datos:
                f.write(dato + '\n\n')
        logging.info(f"Datos guardados en formato texto en {output_file}")
    except Exception as e:
        logging.error(f"Error al guardar los datos en TXT: {e}")

def convertir_a_json(txt_file, json_file):
    """Convierte el archivo TXT con la estructura a un archivo JSON."""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            contenido = f.read()

        # Inicializar el diccionario principal
        datos_json = {}
        seccion_actual = None
        subseccion_actual = None
        temp_dict = {}

        # Procesar el contenido línea por línea
        lineas = contenido.split('\n')
        for linea in lineas:
            linea = linea.strip()

            # Saltar líneas vacías y encabezados de página
            if not linea or "Página" in linea or "Impreso el" in linea:
                continue

            # Detectar secciones principales
            if linea in ["DATOS_BASICOS", "IDENTIFICACION", "EQUIPO_QUIRURGICO",
                        "DIAGNOSTICOS", "INTERVENCION", "PROCEDIMIENTOS",
                        "DESCRIPCION_QUIRURGICA", "CONTEO_MATERIAL", "DATOS_PROFESIONAL"]:
                if seccion_actual and temp_dict:
                    datos_json[seccion_actual] = temp_dict.copy()
                seccion_actual = linea
                temp_dict = {}
                subseccion_actual = None
                continue

            # Procesar líneas con información
            if seccion_actual and ":" in linea:
                clave, valor = [x.strip() for x in linea.split(":", 1)]

                # Manejo especial para DESCRIPCION_QUIRURGICA
                if seccion_actual == "DESCRIPCION_QUIRURGICA":
                    if "Hallazgos_operatorios" in clave:
                        subseccion_actual = "Hallazgos_operatorios"
                        temp_dict[subseccion_actual] = valor
                    elif "Detalle_quirúrgico" in clave:
                        subseccion_actual = "Detalle_quirúrgico"
                        temp_dict[subseccion_actual] = valor
                    elif subseccion_actual:
                        temp_dict[subseccion_actual] += "\n" + linea
                    else:
                        temp_dict[clave] = valor
                else:
                    temp_dict[clave] = valor
            # Continuar acumulando texto para subsecciones
            elif seccion_actual and subseccion_actual and linea:
                temp_dict[subseccion_actual] += "\n" + linea

        # Guardar la última sección
        if seccion_actual and temp_dict:
            datos_json[seccion_actual] = temp_dict

        # Guardar como JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(datos_json, f, indent=4, ensure_ascii=False)

        logging.info(f"Datos convertidos y guardados en formato JSON en {json_file}")

    except Exception as e:
        logging.error(f"Error al convertir los datos a JSON: {e}")

def validar_datos(datos_json):
    """Valida que todas las secciones requeridas estén presentes y completas."""
    secciones_requeridas = [
        "DATOS_BASICOS", "IDENTIFICACION", "EQUIPO_QUIRURGICO",
        "DIAGNOSTICOS", "INTERVENCION", "PROCEDIMIENTOS",
        "DESCRIPCION_QUIRURGICA", "CONTEO_MATERIAL", "DATOS_PROFESIONAL"
    ]

    for seccion in secciones_requeridas:
        if seccion not in datos_json:
            logging.warning(f"Sección {seccion} faltante en el JSON")
        elif seccion == "DESCRIPCION_QUIRURGICA":
            if "Hallazgos_operatorios" not in datos_json[seccion]:
                logging.warning("Hallazgos operatorios faltantes")
            if len(datos_json[seccion]["Hallazgos_operatorios"]) < 50:
                logging.warning("Hallazgos operatorios posiblemente incompletos")

if __name__ == '__main__':
    # Configuración de directorios
    DIRECTORIO_PDFS = "./"
    DIRECTORIO_SALIDA = "./"

    # Crear directorio de salida si no existe
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

    # Procesar informes quirúrgicos
    logging.info("Iniciando procesamiento de informes quirúrgicos...")
    start_time = time.time()

    resultados = []
    for filename in os.listdir(DIRECTORIO_PDFS):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(DIRECTORIO_PDFS, filename)
            logging.info(f"Procesando {filename}...")
            datos = procesar_informe_quirurgico(pdf_path)
            if datos:
                resultados.append(datos)

    # Guardar resultados en TXT
    txt_file = os.path.join(DIRECTORIO_SALIDA, "informes_quirurgicos.txt")
    guardar_en_txt(resultados, txt_file)

    # Convertir resultados a JSON
    json_file = os.path.join(DIRECTORIO_SALIDA, "informes_quirurgicos.json")
    convertir_a_json(txt_file, json_file)

    # Validar el JSON resultante
    with open(json_file, 'r', encoding='utf-8') as f:
        datos_json = json.load(f)
    validar_datos(datos_json)

    end_time = time.time()
    logging.info(f"Proceso completado en {end_time - start_time:.2f} segundos")