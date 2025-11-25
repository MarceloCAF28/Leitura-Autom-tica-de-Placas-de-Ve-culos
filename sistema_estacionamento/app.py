from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import cv2
import numpy as np
import re
import os
import sqlite3
from datetime import datetime, timedelta
import pytesseract
from werkzeug.utils import secure_filename
import subprocess
import base64
import easyocr
import torch

# ========== CONFIGURAÇÃO INICIAL ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, 
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.config['SECRET_KEY'] = 'sistema_estacionamento_secret_key_2024'
app.config['UPLOAD_FOLDER'] = os.path.join(STATIC_DIR, 'uploads')
app.config['DATABASE'] = 'sistema_estacionamento.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ CONFIGURAÇÃO DOS MODELOS OCR ------------------
def configurar_tesseract():
    """Configura apenas o Tesseract de forma robusta"""
    try:
        # Tentar caminhos comuns
        caminhos_tentativos = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "tesseract"  # Tentar usar do PATH
        ]
        
        for caminho in caminhos_tentativos:
            if os.path.exists(caminho) or caminho == "tesseract":
                try:
                    pytesseract.pytesseract.tesseract_cmd = caminho
                    # Testar se funciona
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Tesseract configurado: {caminho} (v{version})")
                    return True
                except:
                    continue
        
        print("❌ Tesseract não encontrado. Instale em: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
        
    except Exception as e:
        print(f"❌ Erro ao configurar Tesseract: {e}")
        return False

print("🔄 Configurando EasyOCR...")
try:
    # Verificar se CUDA está disponível (GPU)
    use_gpu = torch.cuda.is_available()
    print(f"🎯 GPU disponível: {use_gpu}")
    
    # Inicializar EasyOCR para português e inglês
    READER = easyocr.Reader(['pt', 'en'], gpu=use_gpu)
    EASYOCR_CONFIGURADO = True
    print("✅ EasyOCR configurado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao configurar EasyOCR: {e}")
    READER = None
    EASYOCR_CONFIGURADO = False

print("🔄 Configurando Tesseract...")
TESSERACT_CONFIGURADO = configurar_tesseract()

# ------------------ BANCO DE DADOS ------------------
class DatabaseManager:
    def __init__(self):
        self.db_path = app.config['DATABASE']
        self.connection = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            return True
        except Exception as err:
            print(f"❌ Erro de conexão: {err}")
            return False

    def create_tables(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registros_placas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    placa VARCHAR(10) NOT NULL,
                    horario_entrada DATETIME,
                    horario_saida DATETIME,
                    dia DATE,
                    tempo_permanencia TEXT,
                    valor_cobrado DECIMAL(10,2),
                    status TEXT DEFAULT 'estacionado',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configuracoes (
                    id INTEGER PRIMARY KEY,
                    valor_hora DECIMAL(10,2) DEFAULT 5.00,
                    valor_diaria DECIMAL(10,2) DEFAULT 50.00
                )
            ''')
            
            cursor.execute('SELECT * FROM configuracoes WHERE id = 1')
            if not cursor.fetchone():
                cursor.execute('INSERT INTO configuracoes (id, valor_hora, valor_diaria) VALUES (1, 5.00, 50.00)')
            
            self.connection.commit()
            return True
        except Exception as err:
            print(f"❌ Erro ao criar tabelas: {err}")
            return False

    def registrar_entrada(self, placa):
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT id FROM registros_placas 
                WHERE placa = ? AND status = 'estacionado'
            ''', (placa,))
            if cursor.fetchone():
                return False, "⚠️ Veículo já está no estacionamento."
            
            agora = datetime.now()
            cursor.execute('''
                INSERT INTO registros_placas (placa, horario_entrada, dia, status)
                VALUES (?, ?, ?, 'estacionado')
            ''', (placa, agora, agora.date()))
            
            self.connection.commit()
            return True, f"✅ Entrada registrada para placa: {placa}"
        except Exception as err:
            return False, f"❌ Erro ao registrar entrada: {err}"

    def registrar_saida(self, placa):
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT id, horario_entrada FROM registros_placas
                WHERE placa = ? AND status = 'estacionado'
                ORDER BY horario_entrada DESC LIMIT 1
            ''', (placa,))
            
            resultado = cursor.fetchone()
            if not resultado:
                return False, "❌ Nenhuma entrada aberta encontrada para esta placa."
            
            registro_id, horario_entrada = resultado['id'], resultado['horario_entrada']
            agora = datetime.now()
            
            if isinstance(horario_entrada, str):
                try:
                    horario_entrada = datetime.fromisoformat(horario_entrada.replace('Z', '+00:00'))
                except:
                    horario_entrada = datetime.strptime(horario_entrada, "%Y-%m-%d %H:%M:%S")
            
            tempo_permanencia = agora - horario_entrada
            total_segundos = tempo_permanencia.total_seconds()
            horas = total_segundos / 3600
            
            cursor.execute('SELECT valor_hora FROM configuracoes WHERE id = 1')
            valor_hora = cursor.fetchone()['valor_hora']
            valor_total = round(horas * valor_hora, 2)
            
            horas_int = int(total_segundos // 3600)
            minutos_int = int((total_segundos % 3600) // 60)
            segundos_int = int(total_segundos % 60)
            tempo_formatado = f"{horas_int:02d}h {minutos_int:02d}min {segundos_int:02d}s"
            
            cursor.execute('''
                UPDATE registros_placas 
                SET horario_saida = ?, tempo_permanencia = ?, valor_cobrado = ?, status = 'finalizado'
                WHERE id = ?
            ''', (agora, tempo_formatado, valor_total, registro_id))
            
            self.connection.commit()
            
            mensagem = (
                f"✅ Saída registrada para placa: {placa}\n"
                f"⏱️ Tempo: {tempo_formatado}\n"
                f"💰 Valor: R$ {valor_total:.2f}"
            )
            return True, mensagem
            
        except Exception as err:
            return False, f"❌ Erro ao registrar saída: {err}"

    def buscar_registros(self, placa=None, dia=None, tipo_filtro=None, inicio=None, fim=None, limit=None, apenas_estacionados=False):
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM registros_placas WHERE 1=1"
            params = []
            
            if placa:
                query += " AND placa = ?"
                params.append(placa)
            elif dia:
                query += " AND dia = ?"
                params.append(dia)
            elif tipo_filtro and inicio and fim:
                coluna = "horario_entrada" if tipo_filtro == "entrada" else "horario_saida"
                query += f" AND {coluna} IS NOT NULL AND time({coluna}) BETWEEN ? AND ?"
                params.extend([inicio, fim])
            elif apenas_estacionados:
                query += " AND status = 'estacionado'"
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as err:
            print(f"❌ Erro ao buscar registros: {err}")
            return []

    def limpar_historico(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute('DELETE FROM registros_placas')
            self.connection.commit()
            return True
        except Exception as e:
            print(f"❌ Erro ao limpar histórico: {e}")
            return False

    def get_estatisticas(self):
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM registros_placas WHERE status = "estacionado"')
            estacionados = cursor.fetchone()['total']
            
            hoje = datetime.now().date()
            cursor.execute('''
                SELECT COALESCE(SUM(valor_cobrado), 0) as total 
                FROM registros_placas 
                WHERE DATE(horario_saida) = ? AND status = "finalizado"
            ''', (hoje,))
            faturamento_hoje = cursor.fetchone()['total']
            
            cursor.execute('SELECT COUNT(*) as total FROM registros_placas')
            total_registros = cursor.fetchone()['total']
            
            return {
                'estacionados': estacionados,
                'faturamento_hoje': float(faturamento_hoje),
                'total_registros': total_registros
            }
        except Exception as e:
            print(f"❌ Erro ao buscar estatísticas: {e}")
            return {'estacionados': 0, 'faturamento_hoje': 0.0, 'total_registros': 0}

    def get_configuracoes(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM configuracoes WHERE id = 1')
            config = cursor.fetchone()
            if config:
                return {
                    'id': config['id'],
                    'valor_hora': float(config['valor_hora']),
                    'valor_diaria': float(config['valor_diaria'])
                }
            return None
        except:
            return None

    def atualizar_configuracoes(self, valor_hora, valor_diaria):
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE configuracoes 
                SET valor_hora = ?, valor_diaria = ?
                WHERE id = 1
            ''', (valor_hora, valor_diaria))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"❌ Erro ao atualizar configurações: {e}")
            return False

# ------------------ DETECÇÃO MELHORADA PARA PLACAS ANTIGAS E MERCOSUL ------------------
def detectar_regioes_placas(imagem):
    """
    Detecta regiões de placas com técnicas específicas para placas antigas e Mercosul
    """
    try:
        # Converter para escala de cinza
        if len(imagem.shape) == 3:
            gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        else:
            gray = imagem.copy()
        
        altura, largura = gray.shape
        print(f"📐 Dimensões da imagem: {largura}x{altura}")
        
        regioes_placas = []
        
        # MÚLTIPLAS ESTRATÉGIAS DE PRÉ-PROCESSAMENTO
        estrategias = []
        
        # Estratégia 1: CLAHE + Bilateral Filter
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)
        estrategias.append(bilateral)
        
        # Estratégia 2: Equalização de histograma normal
        equalized = cv2.equalizeHist(gray)
        estrategias.append(equalized)
        
        # Estratégia 3: Filtro de mediana para ruído
        median = cv2.medianBlur(gray, 5)
        estrategias.append(median)
        
        for idx, estrategia_img in enumerate(estrategias):
            print(f"🔍 Aplicando estratégia de pré-processamento {idx + 1}")
            
            # DETECÇÃO DE BORDAS COM MÚLTIPLOS PARÂMETROS
            parametros_edges = [
                (30, 100),   # Baixo threshold - mais sensível
                (50, 150),   # Médio threshold
                (70, 200)    # Alto threshold - mais específico
            ]
            
            for low_thresh, high_thresh in parametros_edges:
                edges = cv2.Canny(estrategia_img, low_thresh, high_thresh)
                
                # OPERAÇÕES MORFOLÓGICAS DIFERENTES
                kernels = [
                    cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),   # Retangular horizontal
                    cv2.getStructuringElement(cv2.MORPH_RECT, (12, 6)),   # Retangular vertical
                    cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)),   # Mais estreito
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 5)) # Elíptico
                ]
                
                for kernel in kernels:
                    # Fechamento para conectar componentes da placa
                    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                    
                    # Encontrar contornos
                    contornos, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contorno in contornos:
                        area = cv2.contourArea(contorno)
                        
                        # FILTROS MAIS FLEXÍVEIS PARA PLACAS
                        if area < 1000 or area > 50000:  # Área expandida
                            continue
                            
                        x, y, w, h = cv2.boundingRect(contorno)
                        proporcao = w / h if h > 0 else 0
                        
                        # Proporções características de placas brasileiras
                        # Placa antiga: ~2.5:1, Mercosul: ~2.8:1, algumas distorções: 2.0-5.0
                        if 1.8 < proporcao < 5.5:
                            # Verificar solidez
                            hull = cv2.convexHull(contorno)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0
                            
                            # Calcular extensão (quão retangular é)
                            rect_area = w * h
                            extent = float(area) / rect_area if rect_area > 0 else 0
                            
                            # Critérios mais flexíveis para placas
                            if solidity > 0.6 and extent > 0.5:
                                # Expansão para capturar toda a placa
                                expand_x = int(w * 0.1)  # 10% de expansão
                                expand_y = int(h * 0.15) # 15% de expansão
                                
                                x_new = max(0, x - expand_x)
                                y_new = max(0, y - expand_y)
                                w_new = min(largura - x_new, w + 2 * expand_x)
                                h_new = min(altura - y_new, h + 2 * expand_y)
                                
                                # Verificar se a região não é muito pequena
                                if w_new > 50 and h_new > 20:
                                    regioes_placas.append((x_new, y_new, w_new, h_new))
        
        # REMOVER DUPLICATAS E SOBREPOSIÇÕES
        regioes_filtradas = []
        for regiao in regioes_placas:
            x, y, w, h = regiao
            area = w * h
            
            # Verificar sobreposição com regiões já selecionadas
            sobreposta = False
            for i, (x2, y2, w2, h2) in enumerate(regioes_filtradas):
                area2 = w2 * h2
                
                # Calcular interseção
                x_left = max(x, x2)
                y_top = max(y, y2)
                x_right = min(x + w, x2 + w2)
                y_bottom = min(y + h, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    area_intersecao = (x_right - x_left) * (y_bottom - y_top)
                    area_minima = min(area, area2)
                    
                    # Se há grande sobreposição (>40%), manter a maior
                    if area_intersecao > area_minima * 0.4:
                        if area > area2:
                            regioes_filtradas[i] = regiao
                        sobreposta = True
                        break
            
            if not sobreposta:
                regioes_filtradas.append(regiao)
        
        print(f"🎯 Regiões de placa detectadas: {len(regioes_filtradas)}")
        return regioes_filtradas
        
    except Exception as e:
        print(f"❌ Erro na detecção de placas: {e}")
        return []

def preprocessar_placa_ocr(imagem):
    """
    Pré-processamento ESPECÍFICO para OCR de placas antigas e Mercosul
    """
    try:
        if len(imagem.shape) == 3:
            gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        else:
            gray = imagem.copy()
        
        # 1. Redimensionar para tamanho ideal
        h, w = gray.shape
        if h < 50 or w < 150:
            scale = max(60 / h, 250 / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Múltiplas técnicas de melhoria de contraste
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        
        # Equalização de histograma global
        equalized = cv2.equalizeHist(gray)
        
        # 3. Denoising
        denoised1 = cv2.bilateralFilter(clahe_img, 9, 75, 75)
        denoised2 = cv2.medianBlur(equalized, 3)
        
        # 4. Múltiplas técnicas de binarização
        # Otsu
        _, otsu1 = cv2.threshold(denoised1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu2 = cv2.threshold(denoised2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptativo
        adaptative1 = cv2.adaptiveThreshold(denoised1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        adaptative2 = cv2.adaptiveThreshold(denoised2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 15, 3)
        
        # 5. Operações morfológicas para limpeza
        kernel_clean1 = np.ones((2, 2), np.uint8)
        kernel_clean2 = np.ones((1, 1), np.uint8)
        
        # Aplicar limpeza morfológica
        versoes_limpas = {}
        
        for nome, img in [('otsu1', otsu1), ('otsu2', otsu2), 
                         ('adapt1', adaptative1), ('adapt2', adaptative2)]:
            # Fechamento para unir caracteres quebrados
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_clean1)
            # Abertura para remover ruído
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_clean2)
            versoes_limpas[nome] = opened
        
        versoes_limpas['original'] = gray
        versoes_limpas['clahe'] = clahe_img
        
        return versoes_limpas
        
    except Exception as e:
        print(f"❌ Erro no pré-processamento: {e}")
        return None

def reconhecer_texto_easyocr_otimizado(imagem):
    """
    EasyOCR OTIMIZADO para placas brasileiras
    """
    try:
        if not EASYOCR_CONFIGURADO or READER is None:
            return "EASYOCR_NÃO_CONFIGURADO", 0
        
        # Converter para BGR
        if len(imagem.shape) == 2:
            imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        else:
            imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
        
        # Configurações OTIMIZADAS para placas brasileiras
        results = READER.readtext(
            imagem_bgr,
            decoder='beamsearch',
            batch_size=1,
            workers=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-', # INCLUI HIFEN
            detail=1,
            paragraph=False,
            min_size=20,
            text_threshold=0.5,  # LIMIAR MAIS BAIXO PARA CAPTURAR MAIS
            low_text=0.3,
            link_threshold=0.3,
            width_ths=0.8,
            height_ths=0.8
        )
        
        textos_validos = []
        confiancas_validas = []
        
        for (bbox, text, conf) in results:
            # LIMPEZA E VALIDAÇÃO
            text_clean = text.strip().upper()
            
            # Remover caracteres especiais, mas manter hífen
            text_clean = re.sub(r'[^A-Z0-9-]', '', text_clean)
            
            print(f"    👁️ EasyOCR raw: '{text}' -> '{text_clean}' (conf: {conf:.2f})")
            
            # VALIDAÇÃO DE PLACAS BRASILEIRAS
            # Padrão antigo: LLL-NNNN (7 caracteres com hífen)
            # Padrão Mercosul: LLLNLNN (7 caracteres sem hífen)
            # Ou versões parciais que podemos corrigir
            
            comprimento = len(text_clean.replace('-', ''))
            
            # Deve ter letras E números
            tem_letras = any(c.isalpha() for c in text_clean)
            tem_numeros = any(c.isdigit() for c in text_clean)
            
            if (4 <= comprimento <= 8 and
                tem_letras and tem_numeros and
                conf > 0.3):  # Confiança mais baixa para capturar mais
                
                textos_validos.append(text_clean)
                confiancas_validas.append(conf)
                print(f"    ✅ ACEITO: '{text_clean}' (conf: {conf:.2f}, comprimento: {comprimento})")
            else:
                print(f"    ❌ REJEITADO: '{text_clean}' - comprimento: {comprimento}, letras: {tem_letras}, números: {tem_numeros}, conf: {conf:.2f}")
        
        if textos_validos:
            # Escolher o melhor resultado (maior confiança)
            melhor_idx = np.argmax(confiancas_validas)
            texto_final = textos_validos[melhor_idx]
            confianca_final = confiancas_validas[melhor_idx] * 100
            return texto_final, confianca_final
        else:
            return "NÃO_RECONHECIDO", 0
            
    except Exception as e:
        print(f"❌ Erro no EasyOCR: {e}")
        return "ERRO_EASYOCR", 0

def reconhecer_texto_tesseract_otimizado(imagem):
    """
    Tesseract otimizado para placas brasileiras
    """
    try:
        if not TESSERACT_CONFIGURADO:
            return "TESSERACT_NÃO_CONFIGURADO", 0
        
        # Configurações específicas para placas
        configs = [
            "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        ]
        
        melhores_resultados = []
        
        for config in configs:
            try:
                dados = pytesseract.image_to_data(imagem, config=config, output_type=pytesseract.Output.DICT)
                
                textos_detectados = []
                confiancas_detectadas = []
                
                for i, texto in enumerate(dados['text']):
                    texto_limpo = texto.strip()
                    confianca = dados['conf'][i]
                    
                    if (texto_limpo and 
                        len(texto_limpo) >= 2 and 
                        confianca > 30 and  # Confiança mais baixa
                        all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in texto_limpo)):
                        
                        textos_detectados.append(texto_limpo)
                        confiancas_detectadas.append(confianca)
                
                if textos_detectados:
                    texto_completo = ''.join(textos_detectados)
                    confianca_media = np.mean(confiancas_detectadas) if confiancas_detectadas else 0
                    
                    # Validação de placa brasileira
                    texto_sem_hifen = texto_completo.replace('-', '')
                    tem_letras = any(c.isalpha() for c in texto_sem_hifen)
                    tem_numeros = any(c.isdigit() for c in texto_sem_hifen)
                    
                    if (4 <= len(texto_sem_hifen) <= 8 and
                        tem_letras and tem_numeros):
                        
                        melhores_resultados.append((texto_completo, confianca_media))
                        print(f"    ✅ Tesseract {config}: '{texto_completo}' (conf: {confianca_media:.1f})")
                    
            except Exception as e:
                print(f"    ⚠️ Tesseract erro com {config}: {e}")
                continue
        
        if melhores_resultados:
            melhor = max(melhores_resultados, key=lambda x: x[1])
            return melhor[0].upper(), melhor[1]
        else:
            return "NÃO_RECONHECIDO", 0
            
    except Exception as e:
        print(f"❌ Erro no Tesseract: {e}")
        return "ERRO_TESSERACT", 0

def formatar_placa_corretamente(texto):
    """
    Formata placa brasileira CORRETAMENTE para modelos antigos e Mercosul
    """
    try:
        if not texto or texto in ["NÃO_RECONHECIDO", "ERRO_TESSERACT", "ERRO_EASYOCR"]:
            return "PLACA-NÃO-LIDA"
        
        # Limpar e padronizar
        texto_limpo = re.sub(r'[^A-Z0-9]', '', texto.upper())
        
        if len(texto_limpo) < 4:
            return "PLACA-INVALIDA"
        
        # Separar letras e números
        letras = ''.join([c for c in texto_limpo if c.isalpha()])
        numeros = ''.join([c for c in texto_limpo if c.isdigit()])
        
        print(f"🔤 Formatação: texto='{texto}' -> limpo='{texto_limpo}' -> letras='{letras}', números='{numeros}'")
        
        # TENTAR IDENTIFICAR PADRÃO MERCOSUL PRIMEIRO (LLLNLNN)
        if len(letras) >= 4 and len(numeros) >= 3:
            # Padrão: 3 letras + 1 número + 1 letra + 2 números
            padrao_mercosul = f"{letras[:3]}{numeros[0]}{letras[3] if len(letras) > 3 else 'A'}{numeros[1:3]}"
            if len(padrao_mercosul) == 7:
                placa_formatada = f"{padrao_mercosul[:3]}{padrao_mercosul[3]}{padrao_mercosul[4]}-{padrao_mercosul[5:7]}"
                print(f"  🆕 Formato Mercosul detectado: {placa_formatada}")
                return placa_formatada
        
        # PADRÃO ANTIGO (LLL-NNNN)
        letras_final = letras[:3] if letras else "XXX"
        numeros_final = numeros[:4] if numeros else "0000"
        
        # Garantir comprimento correto
        if len(letras_final) < 3:
            letras_final = letras_final.ljust(3, 'X')
        if len(numeros_final) < 4:
            numeros_final = numeros_final.ljust(4, '0')
        
        placa_final = f"{letras_final}-{numeros_final}"
        
        # VALIDAÇÃO FINAL - Rejeitar padrões inválidos
        padroes_invalidos = [
            'WWW-0000', 'XXX-0000', 'AAA-0000', 'ABC-1234',
            'TEST-0000', 'NOR-0000', 'PLACA-NÃO-LIDA', 'PLACA-INVALIDA',
            'XXXX-0000', 'XXXX-000', 'XXX-000'
        ]
        
        if placa_final in padroes_invalidos:
            return "PLACA-INVALIDA"
        
        print(f"  🆗 Formato antigo aplicado: {placa_final}")
        return placa_final
        
    except Exception as e:
        print(f"⚠️ Erro na formatação: {e}")
        return "ERRO-FORMATACAO"

def processar_imagem_placas(file):
    """
    Processamento OTIMIZADO para placas brasileiras
    """
    try:
        # Ler imagem
        file_bytes = np.frombuffer(file.read(), np.uint8)
        imagem = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if imagem is None:
            return False, "Erro: Não foi possível ler a imagem"
        
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem_original = imagem_rgb.copy()
        
        print("=" * 60)
        print("🔄 PROCESSAMENTO DE PLACAS - DETECÇÃO MELHORADA")
        print("=" * 60)
        
        # 1. Detectar regiões de placa com múltiplas estratégias
        regioes = detectar_regioes_placas(imagem_rgb)
        
        resultados = []
        
        # 2. Processar cada região detectada
        for i, (x, y, w, h) in enumerate(regioes):
            print(f"\n🔍 Analisando região de placa {i+1}/{len(regioes)}")
            print(f"   📍 Posição: ({x}, {y}) Tamanho: {w}x{h}")
            
            roi = imagem_rgb[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # PRÉ-PROCESSAMENTO AVANÇADO
            versoes = preprocessar_placa_ocr(roi)
            
            if not versoes:
                continue
            
            # TENTAR MÚLTIPLAS COMBINAÇÕES DE OCR
            melhor_texto = "NÃO_RECONHECIDO"
            melhor_confianca = 0
            melhor_metodo = "NENHUM"
            
            for nome, versao_imagem in versoes.items():
                if versao_imagem is not None:
                    # EasyOCR primeiro (geralmente melhor para placas)
                    texto_easy, conf_easy = reconhecer_texto_easyocr_otimizado(versao_imagem)
                    
                    if (texto_easy not in ["NÃO_RECONHECIDO", "ERRO_EASYOCR"] and 
                        conf_easy > melhor_confianca):
                        melhor_texto = texto_easy
                        melhor_confianca = conf_easy
                        melhor_metodo = f"EasyOCR-{nome}"
                        print(f"   🎯 Novo melhor (EasyOCR): '{melhor_texto}' ({melhor_confianca:.1f}%)")
                    
                    # Tesseract como fallback
                    if melhor_confianca < 50:  # Se EasyOCR não foi bom o suficiente
                        texto_tess, conf_tess = reconhecer_texto_tesseract_otimizado(versao_imagem)
                        if (texto_tess not in ["NÃO_RECONHECIDO", "ERRO_TESSERACT"] and 
                            conf_tess > melhor_confianca):
                            melhor_texto = texto_tess
                            melhor_confianca = conf_tess
                            melhor_metodo = f"Tesseract-{nome}"
                            print(f"   🎯 Novo melhor (Tesseract): '{melhor_texto}' ({melhor_confianca:.1f}%)")
            
            # Formatar placa
            placa_formatada = formatar_placa_corretamente(melhor_texto)
            
            # Aceitar placas com confiança razoável ou formato válido
            if (placa_formatada not in ["PLACA-INVALIDA", "ERRO-FORMATACAO", "PLACA-NÃO-LIDA"] and
                (melhor_confianca > 30 or 
                 (len(placa_formatada) == 8 and placa_formatada[3] == '-'))):  # Formato válido
                
                print(f"   ✅ PLACA DETECTADA: '{melhor_texto}' -> '{placa_formatada}'")
                print(f"   📊 Confiança: {melhor_confianca:.1f}% | Método: {melhor_metodo}")
                
                # Determinar cor baseado na confiança
                if melhor_confianca > 65:
                    cor = (0, 255, 0)  # Verde - Alta confiança
                    status = "ALTA CONFIANÇA"
                elif melhor_confianca > 40:
                    cor = (255, 255, 0)  # Amarelo - Confiança média
                    status = "CONFIANÇA MÉDIA"
                else:
                    cor = (255, 165, 0)  # Laranja - Baixa confiança mas formato válido
                    status = "FORMATO VÁLIDO"
                
                # Marcar na imagem
                cv2.rectangle(imagem_original, (x, y), (x+w, y+h), cor, 3)
                cv2.putText(imagem_original, placa_formatada, 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
                cv2.putText(imagem_original, f"{melhor_confianca:.1f}% - {status}", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
                
                resultados.append({
                    'placa': placa_formatada,
                    'confianca': float(melhor_confianca),
                    'valido': bool(melhor_confianca > 40 or placa_formatada not in ["PLACA-INVALIDA", "PLACA-NÃO-LIDA"]),
                    'status': status,
                    'metodo': melhor_metodo,
                    'regiao': f"{w}x{h}"
                })
        
        # 3. FALLBACK: Buscar texto em toda a imagem se nenhuma região foi detectada
        if not resultados:
            print("\n🔍 Nenhuma região de placa detectada. Buscando texto na imagem completa...")
            
            versoes_completa = preprocessar_placa_ocr(imagem_rgb)
            
            for nome, versao_imagem in versoes_completa.items():
                if versao_imagem is not None:
                    texto_easy, conf_easy = reconhecer_texto_easyocr_otimizado(versao_imagem)
                    
                    if texto_easy not in ["NÃO_RECONHECIDO", "ERRO_EASYOCR"] and conf_easy > 25:
                        placa_formatada = formatar_placa_corretamente(texto_easy)
                        
                        if placa_formatada not in ["PLACA-INVALIDA", "ERRO-FORMATACAO"]:
                            print(f"📝 Placa encontrada em OCR geral: '{texto_easy}' -> '{placa_formatada}'")
                            
                            resultados.append({
                                'placa': placa_formatada,
                                'confianca': float(conf_easy),
                                'valido': bool(conf_easy > 30),
                                'status': 'OCR GERAL',
                                'metodo': f'EasyOCR-{nome}',
                                'regiao': 'Imagem completa'
                            })
                            
                            # Marcar no centro da imagem
                            h, w = imagem_original.shape[:2]
                            cv2.putText(imagem_original, f"PLACA: {placa_formatada}", 
                                       (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            break
        
        # 4. Resultado final
        if not resultados:
            print("⚠️ Nenhuma placa válida detectada.")
            resultados.append({
                'placa': 'PLACA-NÃO-LIDA',
                'confianca': 0.0,
                'valido': False,
                'status': 'NÃO DETECTADA',
                'metodo': 'Sistema',
                'regiao': 'N/A'
            })
        
        # Converter para base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(imagem_original, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        resultado_final = {
            'imagem': f"data:image/jpeg;base64,{img_str}",
            'placas_detectadas': int(len([r for r in resultados if r['valido']])),
            'detalhes': resultados
        }
        
        print("\n" + "=" * 60)
        print(f"✅ PROCESSAMENTO CONCLUÍDO")
        print(f"📊 Placas válidas: {resultado_final['placas_detectadas']}")
        for i, resultado in enumerate(resultados):
            valido = "✅" if resultado['valido'] else "❌"
            print(f"   {i+1}. {resultado['placa']} {valido} (Conf: {resultado['confianca']:.1f}%) - {resultado['status']}")
        print("=" * 60)
        
        return True, resultado_final
        
    except Exception as e:
        print(f"❌ ERRO NO PROCESSAMENTO: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Erro: {str(e)}"

# ------------------ ROTAS FLASK ------------------
db = DatabaseManager()

@app.before_request
def before_request():
    if not db.connection:
        if not db.connect():
            return "Erro de conexão com o banco de dados", 500
        db.create_tables()

@app.route('/')
def index():
    try:
        filtro = request.args.get('filtro', 'todos')
        placa_busca = request.args.get('placa', '')
        dia_busca = request.args.get('dia', '')
        aba_atual = request.args.get('aba', 'tab-dashboard')
        
        if placa_busca:
            registros = db.buscar_registros(placa=placa_busca, limit=50)
        elif dia_busca:
            registros = db.buscar_registros(dia=dia_busca)
        elif filtro == 'estacionados':
            registros = db.buscar_registros(apenas_estacionados=True)
        else:
            registros = db.buscar_registros(limit=20)
        
        estatisticas = db.get_estatisticas()
        configuracao = db.get_configuracoes()
        
        return render_template('index.html', 
                             estatisticas=estatisticas,
                             registros=registros,
                             configuracao=configuracao,
                             easyocr_configurado=EASYOCR_CONFIGURADO,
                             tesseract_configurado=TESSERACT_CONFIGURADO,
                             now=datetime.now(),
                             filtro_atual=filtro,
                             placa_busca=placa_busca,
                             dia_busca=dia_busca,
                             aba_atual=aba_atual)
    except Exception as e:
        print(f"❌ Erro na rota principal: {e}")
        return "Erro interno do servidor", 500

@app.route('/processar_imagem', methods=['POST'])
def processar_imagem_route():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Nenhum arquivo enviado'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Nenhum arquivo selecionado'})
        
        success, resultado = processar_imagem_placas(file)
        
        if success:
            # Garantir serialização JSON
            if 'detalhes' in resultado:
                for detalhe in resultado['detalhes']:
                    if 'valido' in detalhe:
                        detalhe['valido'] = bool(detalhe['valido'])
                    if 'confianca' in detalhe:
                        detalhe['confianca'] = float(detalhe['confianca'])
            
            return jsonify({
                'success': True, 
                'data': resultado
            })
        else:
            return jsonify({
                'success': False, 
                'message': str(resultado)
            })
            
    except Exception as e:
        print(f"❌ Erro na rota processar_imagem: {e}")
        return jsonify({
            'success': False, 
            'message': f'Erro interno: {str(e)}'
        })

@app.route('/registrar_entrada', methods=['POST'])
def registrar_entrada_route():
    try:
        placa = request.form.get('placa', '').upper().replace(' ', '')
        
        if not placa:
            flash('Placa é obrigatória', 'error')
            return redirect('/')
        
        success, message = db.registrar_entrada(placa)
        
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
        
        return redirect('/?aba=tab-dashboard')
    except Exception as e:
        print(f"❌ Erro na rota registrar_entrada: {e}")
        flash('Erro interno do sistema', 'error')
        return redirect('/')

@app.route('/registrar_saida', methods=['POST'])
def registrar_saida_route():
    try:
        placa = request.form.get('placa', '').upper().replace(' ', '')
        
        if not placa:
            flash('Placa é obrigatória', 'error')
            return redirect('/')
        
        success, message = db.registrar_saida(placa)
        
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
        
        return redirect('/?aba=tab-dashboard')
    except Exception as e:
        print(f"❌ Erro na rota registrar_saida: {e}")
        flash('Erro interno do sistema', 'error')
        return redirect('/')

@app.route('/limpar_historico', methods=['POST'])
def limpar_historico_route():
    try:
        if db.limpar_historico():
            flash('✅ Histórico apagado com sucesso!', 'success')
        else:
            flash('❌ Erro ao apagar histórico', 'error')
        
        return redirect('/?aba=tab-config')
    except Exception as e:
        print(f"❌ Erro na rota limpar_historico: {e}")
        flash('Erro interno do sistema', 'error')
        return redirect('/')

@app.route('/atualizar_configuracoes', methods=['POST'])
def atualizar_configuracoes_route():
    try:
        valor_hora = float(request.form.get('valor_hora', 5.00))
        valor_diaria = float(request.form.get('valor_diaria', 50.00))
        
        if db.atualizar_configuracoes(valor_hora, valor_diaria):
            flash('✅ Configurações atualizadas com sucesso!', 'success')
        else:
            flash('❌ Erro ao atualizar configurações', 'error')
        
        return redirect('/?aba=tab-config')
    except Exception as e:
        print(f"❌ Erro na rota atualizar_configuracoes: {e}")
        flash('Erro interno do sistema', 'error')
        return redirect('/')

@app.route('/api/estatisticas')
def api_estatisticas():
    try:
        estatisticas = db.get_estatisticas()
        return jsonify(estatisticas)
    except Exception as e:
        print(f"❌ Erro na rota api_estatisticas: {e}")
        return jsonify({'error': 'Erro interno'}), 500

@app.route('/status_sistema')
def status_sistema():
    return jsonify({
        'easyocr': bool(EASYOCR_CONFIGURADO),
        'tesseract': bool(TESSERACT_CONFIGURADO),
        'sistema': 'ativo',
        'versao': '5.0-deteccao-melhorada'
    })

@app.route('/testar_ocr')
def testar_ocr():
    """Rota para testar o OCR com uma imagem específica"""
    return '''
    <form action="/processar_imagem" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Testar OCR">
    </form>
    '''

if __name__ == '__main__':
    try:
        if db.connect():
            db.create_tables()
            print("=" * 60)
            print("🚀 SISTEMA DE ESTACIONAMENTO INICIADO COM SUCESSO!")
            print("=" * 60)
            print(f"📊 EasyOCR: {'✅ CONFIGURADO' if EASYOCR_CONFIGURADO else '❌ NÃO CONFIGURADO'}")
            print(f"📊 Tesseract: {'✅ CONFIGURADO' if TESSERACT_CONFIGURADO else '❌ NÃO CONFIGURADO'}")
            print(f"🎯 GPU: {'✅ DISPONÍVEL' if torch.cuda.is_available() else '❌ CPU MODE'}")
            print("🎯 Modo: DETECÇÃO MELHORADA PARA PLACAS BRASILEIRAS")
            print("🔍 Alvo: Modelos antigos (LLL-NNNN) e Mercosul (LLLNLNN)")
            print("📝 Estratégia: Múltiplos pré-processamentos + Validação inteligente")
            print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ ERRO AO INICIAR SISTEMA: {e}")