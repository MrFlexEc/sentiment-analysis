# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:35:16 2024

@author: Grupo-7
"""
# 1 Instalar librerias
#pip install tweepy textblob wordcloud pandas numpy matplotlib
#pip install selenium
#pip install twarc2 wordcloud nltk spacy
#python -m spacy download es_core_news_sm


# 2 Importación de librerias
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist, bigrams, trigrams
from twarc import Twarc2, expansions
from nltk import ngrams
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob 
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#web scraping con selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time




#3 Extracción de los datos

# 3 Extracción de los datos

# Configurar opciones para el navegador Chrome
chrome_options = Options()
#chrome_options.add_argument('--headless')  # Para ejecutar el navegador en modo headless (sin interfaz gráfica)
chrome_options.add_argument('--no-sandbox')  # Opción adicional para configurar Chrome en entornos Docker, por ejemplo

# Ruta al ejecutable de ChromeDriver (cambia esta ruta según donde hayas descargado ChromeDriver)
chrome_driver_path = 'C:\Mineria_de_Texto\chromedriver-win64/chromedriver.exe'

# Iniciar el servicio de ChromeDriver
service = Service(chrome_driver_path)

# Iniciar el navegador Chrome con Selenium
driver = webdriver.Chrome(service=service, options=chrome_options)

# Navegar a la página de inicio de sesión de X
driver.get('https://twitter.com/login')

# Especificar el tema que deseas rastrear
buscar = '(Estado OR Militares OR corrupción OR terrorismo OR ataques OR enemigos OR bandas OR robos OR delincuencias OR noboa OR armas OR disparos OR explosivos OR tanques OR policias OR atentados OR amenazas OR muertes OR secuestros)  geocode:-1.16017132420716,-78.4329147227382,10.4km lang:es'

# Esperar a que aparezca el campo de nombre de usuario
username_field = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="text"][autocomplete="username"]'))
)

# Ingresar el nombre de usuario
username_field.send_keys("tu correo o nombre de usuario")
# Encontrar y hacer clic en el botón "Siguiente" para iniciar sesión
login_button = driver.find_element(By.XPATH, '//span[contains(text(), "Siguiente")]')
login_button.click()
time.sleep(2)

# Detener el programa y esperar a que se ingrese "seguir" en la consola
while True:
    seguir = input("Escribe 'seguir' para continuar: ")
    if seguir.lower() == "seguir":
        break



# Encontrar el campo de contraseña
password_input = driver.find_element(By.XPATH, '//input[@autocomplete="current-password"]')

# Completar el campo de contraseña con tu valor deseado
password_input.send_keys("tu contraseña del correo")
time.sleep(3)

# Encontrar el botón "Iniciar sesión"
password_input.send_keys(Keys.ENTER)

# Esperar a que se inicie sesión correctamente
WebDriverWait(driver, 3).until(
    EC.url_contains('https://x.com/home')
)

# Navegar a la página de búsqueda de Twitter para el texto de búsqueda dado
driver.get(f'https://twitter.com/search?q={buscar}&src=typed_query&f=live')
time.sleep(3)

# Obtener la altura actual de la ventana del navegador
last_height = driver.execute_script("return document.body.scrollHeight")

# Desplazarse hacia abajo un máximo de 60 veces
scroll_count = 0
max_scroll_count = 60

tweet_texts = []  # Lista para almacenar los textos de los tweets



while scroll_count < max_scroll_count:
    print(f'scroll: {scroll_count}')
    # Desplazarse hacia abajo
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Esperar un breve tiempo para que la página se cargue después de desplazarse
    time.sleep(2)
    
    # Encontrar los elementos <div> con el atributo data-testid="tweetText"
    div_elements = driver.find_elements("xpath", '//div[@data-testid="tweetText"]')
    
    # Recorrer los elementos y agregar los textos a la lista
    for div_element in div_elements:
        tweet_texts.append(div_element.text)
    
    # Incrementar el contador de desplazamiento
    scroll_count += 1

# Borrar tweets duplicados
tweet_texts = list(set(tweet_texts))  # Eliminar duplicados


print(tweet_texts)  # Imprimir la lista sin duplicados

# Definir la ruta y el nombre del archivo CSV
carpeta_destino = 'C:\Mineria_de_Texto'  # Cambia esta ruta por la ruta de la carpeta donde deseas guardar el archivo CSV
nombre_archivo = 'tweets_recopilados_estado.csv'

# Escribir los datos en el archivo CSV
ruta_completa = f'{carpeta_destino}/{nombre_archivo}'
with open(ruta_completa, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Tweets'])  # Escribir encabezado opcional
    for tweet in tweet_texts:
        writer.writerow([tweet])

print(f'Se ha guardado el archivo CSV en: {ruta_completa}')



# 4 Seleccionar los datos

# Ruta al archivo CSV que contiene los datos
ruta_archivo_csv = 'C:/Mineria_de_Texto/tweets_recopilados_estado.csv'  

# Cargar el archivo CSV en un DataFrame de Pandas
df = pd.read_csv(ruta_archivo_csv, encoding='utf-8')

# Mostrar los primeros datos del DataFrame para verificar
print("Primeras filas del DataFrame:")
print(df.head())

#df['Clean_Tweets'] = df['Tweets'].apply(clean_text)

# Descargar recursos de nltk
nltk.download('stopwords')


# 5 limpieza de datos 
# Diccionario para normalización de palabras comunes
diccionario_normalizacion = {
    "q": "que",
    "xq": "porque",
    "d": "de",
    "m": "me",
    "t": "te",
    "u": "tú",
    "p": "para",
    "k": "que",
    "x": "por",
    "c/": "con",
    "s/": "sin",
    "tngo": "tengo",
    "s": "es",
    "pasahe": "pasaje",
    "tb": "también",
    "tqm": "te quiero mucho",
    "bno": "bueno",
    "favs": "favoritos",
    "sta": "está",
    "stoy": "estoy",
    "toa": "toda",
    "to": "todo",
    ":)": ""
}
# Función para limpiar el texto de los tweets
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Eliminar menciones
    text = re.sub(r'#', '', text)  # Eliminar el símbolo '#'
    text = re.sub(r'RT[\s]+', '', text)  # Eliminar 'RT'
    text = re.sub(r'https?:\/\/\S+', '', text)  # Eliminar enlaces
    text = re.sub(r'[^a-záéíóúüñ ]', '', text)  # Eliminar caracteres no alfabéticos
    text = ' '.join([word for word in text.split() if word not in stopwords.words('spanish')])  # Eliminar stopwords
    text = ' '.join([diccionario_normalizacion.get(word, word) for word in text.split()])  # Normalizar palabras
    text = ' '.join(dict.fromkeys(text.split()))  # Eliminar palabras repetidas
    # Eliminar palabras específicas
    text = re.sub(r'\bq\b', '', text)  # Eliminar la palabra 'que'
    text = re.sub(r'\bbuena\b', '', text)  # Eliminar la palabra 'buena'
    text = re.sub(r'\bjajaja\b', '', text)  # Eliminar la palabra 'jajaja'

    return text.strip()  # Eliminar espacios en blanco al inicio y al final
"""
# Crear una columna adicional para almacenar los tweets limpios
df['Clean_Tweets'] = df['Tweets'].apply(clean_text)
# Mostrar los primeros 5 tweets originales y limpios
print(df[['Tweets', 'Clean_Tweets']].head(1))
"""

# Aplicar la limpieza de texto a los tweets
df['Tweets'] = df['Tweets'].apply(clean_text)

# Mostrar los primeros 5 tweets 
print(df.head())

# 6 Modelado

#Nube de palabras
# Generar una nube de palabras
todas_palabras = ' '.join([tweet for tweet in df['Tweets']])
wordcloud = WordCloud(width=900, height=600, random_state=20, max_font_size=110).generate(todas_palabras)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Palabras repetidas

# Obtener todas las palabras de los tweets limpios
palabras_repetidas = ' '.join(df['Tweets'])

# Contar la frecuencia de cada palabra
palabra_frecuencia = Counter(palabras_repetidas.split())

# Convertir a DataFrame de pandas para facilitar la visualización
df_palabra_frecuencia = pd.DataFrame(palabra_frecuencia.items(), columns=['Palabra', 'Frecuencia'])

# Ordenar por frecuencia descendente
df_palabra_frecuencia = df_palabra_frecuencia.sort_values(by='Frecuencia', ascending=False)

# Mostrar los primeros registros para verificar
print(df_palabra_frecuencia.head())

# Visualización en un gráfico de barras de las 20 palabras más frecuentes
plt.figure(figsize=(12, 8))  # Aumentar el tamaño de la figura
df_word_freq_top = df_palabra_frecuencia.head(16)  # Tomar las 16 palabras más frecuentes
bar_plot = df_word_freq_top.plot(kind='barh', x='Palabra', y='Frecuencia', color='skyblue')  # Gráfico de barras horizontal
plt.title('Palabras más frecuentes en Tweets de de Terrorismo en Ecuador')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra')
plt.gca().invert_yaxis()  # Invertir el eje y para ordenar de mayor a menor frecuencia

# Agregar etiquetas de valores en los lados de las columnas
for index, value in enumerate(df_word_freq_top['Frecuencia']):
    plt.text(value, index, str(value))

plt.tight_layout()  # Ajustar el diseño para mejorar la legibilidad
plt.show()



# 7 Unigrama, bigramas y trigramas

# Tokenización de los tweets en palabras individuales
tokenized_tweets = df['Tweets'].str.split()

# Obtener los 10 unigramas más frecuentes
unigrams = [word for tweet in tokenized_tweets for word in tweet]
freq_dist_unigrams = FreqDist(unigrams)
top_unigrams = freq_dist_unigrams.most_common(10)


# Obtener los 10 bigramas más frecuentes
bigram_list = list(bigrams(unigrams))
freq_dist_bigrams = FreqDist(bigram_list)
top_bigrams = freq_dist_bigrams.most_common(10)


# Obtener los 10 trigramas más frecuentes
trigram_list = list(trigrams(unigrams))
freq_dist_trigrams = FreqDist(trigram_list)
top_trigrams = freq_dist_trigrams.most_common(10)

# Convertir a DataFrames de pandas para facilitar la visualización
df_top_unigrams = pd.DataFrame(top_unigrams, columns=['Unigrama', 'Frecuencia'])
df_top_bigrams = pd.DataFrame(top_bigrams, columns=['Bigrama', 'Frecuencia'])
df_top_trigrams = pd.DataFrame(top_trigrams, columns=['Trigrama', 'Frecuencia'])

print("Top 10 Unigramas:")
print(df_top_unigrams)
print("\nTop 10 Bigramas:")
print(df_top_bigrams)
print("\nTop 10 Trigramas:")
print(df_top_trigrams)


plt.figure(figsize=(12, 6))

# Gráfico de barras horizontales para los 10 Unigramas más frecuentes
plt.subplot(1, 3, 1)
df_top_unigrams.sort_values(by='Frecuencia').plot(kind='barh', x='Unigrama', y='Frecuencia', color='skyblue', legend=False)
plt.title('Top 10 Unigramas')
plt.xlabel('Frecuencia')
plt.ylabel('Unigrama')

# Gráfico de barras horizontales para los 10 Bigramas más frecuentes
plt.subplot(1, 3, 2)
df_top_bigrams.sort_values(by='Frecuencia').plot(kind='barh', x='Bigrama', y='Frecuencia', color='green', legend=False)
plt.title('Top 10 Bigramas')
plt.xlabel('Frecuencia')
plt.ylabel('Bigrama')

# Gráfico de barras horizontales para los 10 Trigramas más frecuentes
plt.subplot(1, 3, 3)
df_top_trigrams.sort_values(by='Frecuencia').plot(kind='barh', x='Trigrama', y='Frecuencia', color='orange', legend=False)
plt.title('Top 10 Trigramas')
plt.xlabel('Frecuencia')
plt.ylabel('Trigrama')

plt.tight_layout()
plt.show()


#Vader

nltk.download('vader_lexicon')

# Inicializar sentimientos VADER
sid = SentimentIntensityAnalyzer()
# Función para obtener el puntaje de sentimiento compuesto con VADER
def get_vader_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']
# Aplicar VADER a los tweets limpios
df['Sentimiento_VADER'] = df['Tweets'].apply(get_vader_sentiment)

# Función para convertir el puntaje de sentimiento compuesto de VADER en etiquetas personalizadas
def categorizar_sentimiento(score):
    if score >= 0.05:
        return 'Enojado'
    elif score <= -0.05:
        return 'Triste'
    else:
        return 'Normal'

# Aplicar la función para categorizar sentimientos a los puntajes de VADER
df['Sentimiento_Categorizado'] = df['Sentimiento_VADER'].apply(categorizar_sentimiento)

# Mostrar los primeros 5 tweets con sus puntajes de sentimiento categorizados
print(df[['Tweets', 'Sentimiento_VADER', 'Sentimiento_Categorizado']].head())

# Contar la frecuencia de cada categoría de sentimiento
sentiment_counts = df['Sentimiento_Categorizado'].value_counts()

# Crear un gráfico de barras para mostrar la distribución de sentimientos categorizados
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['red', 'blue', 'green'])
plt.title('Distribución de Emoción')
plt.xlabel('Sentimiento')
plt.ylabel('Número de Tweets')

# Mostrar el número exacto de tweets en cada barra
for i, count in enumerate(sentiment_counts):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

# Mostrar las etiquetas personalizadas en el eje x
plt.xticks(np.arange(3), ['Triste', 'Normal', 'Enojado'], rotation=0)

plt.show()




