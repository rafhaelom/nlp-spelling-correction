########## Corretor Ortográfico utilizando texto de notícias de jornal da web ########## 
# Disciplina: Tópicos Avançados em Ciência de Dados e Inteligência Artificial - NLP.
# Professor: Alexandre Vaz Roriz.
# Autor: Rafhael de Oliveira Martins

## Referências utilizadas
# https://github.com/ViniViniAntunes/Corretor_Ortografico_NLP
# https://towardsdatascience.com/build-a-spelling-corrector-program-in-python-46bc427cf57f
# https://huggingface.co/oliverguhr/spelling-correction-english-base?text=ze+shop+is+cloed+due+to+covid+19


# Importa Bibliotecas.
import streamlit as st

import time
import re

import pandas as pd
import numpy as np

# import psycopg2
# import sqlalchemy  
# from sqlalchemy import create_engine

# from textblob import Word
# from textblob import TextBlob

import nltk
#nltk.download('stopwords')
#stopwords = nltk.corpus.stopwords.words('portuguese')

# Parametros do Pandas, limitando a quantidade máxima e a largura das colunas.
# pd.set_option('max_colwidth', 5000)

# Sidbar menu
st.sidebar.title("Corretor Ortográfico utilizando notícias de jornal web")

##### INÍCIO APP.
st.title('SpellingNW :memo:')
st.markdown("""<p align='justify'>Um Corretor Ortográfico utilizando texto de notícias de jornal da web.<p align='justify'>""", unsafe_allow_html=True)

path = "dados/"

########## Base de dados (Corpus).
# Abrindo o arquivo txt, armazenando o seu conteúdo em uma variável.
with open(path+"conteudo_noticias.txt", mode='r', encoding='utf-8') as f:
    conteudo_treino = f.read()

# lowercase - passa o corpus para letras minúsculas.
conteudo_lower = conteudo_treino.lower()
#print("Quantidade de caracteres no corpus: [", len(conteudo_lower), "]")

# retira pontuações do corpus, permanecendo somentes as palavras.
conteudo_sem_pontuacao = re.sub(r'[^\w\s]', ' ', conteudo_lower)
#print("Quantidade de caracteres no corpus: [", len(conteudo_sem_pontuacao), "]")

# tokenize o corpus por palavras.
conteudo_tokens = re.findall(r'\w+', conteudo_sem_pontuacao)

# Métricas da base.
qtd_caracter_corpus = len(conteudo_treino)
qtd_palavra_corpus = len(conteudo_tokens)
qtd_vocabulario_corpus = len(set(conteudo_tokens))

st.sidebar.write("Total de caracteres no corpus: [", qtd_caracter_corpus, "]")
st.sidebar.write("Total de palvras no corpus: [", qtd_palavra_corpus, "]")
st.sidebar.write("Total de palvras unicas no corpus (vocabulário): [", qtd_vocabulario_corpus, "]")

st.sidebar.markdown("""<p align='justify'>Web App feito por Rafhael de Oliveira Martins<p align='justify'>""", unsafe_allow_html=True)
st.sidebar.write("09 de Dezembro de 2022")
st.sidebar.write("[![GitHub](https://img.shields.io/badge/-GitHub-333333?style=for-the-badge&logo=github)](https://github.com/rafhaelom)" " " "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafhael-martins-3bab63138)")

st.sidebar.markdown("""## Referências: 
* https://github.com/ViniViniAntunes/Corretor_Ortografico_NLP
* https://towardsdatascience.com/build-a-spelling-corrector-program-in-python-46bc427cf57f
* https://huggingface.co/oliverguhr/spelling-correction-english-base?text=ze+shop+is+cloed+due+to+covid+19
""")

########################### MODEL 1 ###########################

##### Calculo da frequência que determinada palavra aparece dentro do corpus. Usando a função FreqDist() da biblioteca nltk. 
# Calcula a frequencia por palavra.
frequencia = nltk.FreqDist(conteudo_tokens)

# Calculando o total de palavras.
total_palavras = len(conteudo_tokens)

# Mostrando as 10 palavras mais comuns da nossa lista_normalizada
frequencia.most_common(10)

### Função probabilidade_palavra.
def probabilidade_palavra(palavra_gerada):
    """
    Função para calcular a probabilidade de determinada palavra aparecer em nosso corpus.
    """
    # Retorna a probabilidade de determinada palavra aparecer no nosso corpus
    return frequencia[palavra_gerada] / total_palavras

### Função insere_letras_faltantes.
def insere_letras_faltantes(palavra_fatiada):
    """
    Função para inserir letra ou letras faltantes na palavra informada.
    Recebe uma lista de tuplas (esquerdo, direito) que corresponde aos lados
    esquerdo e direito da palavra fatiada em dois.
    """

    # Lista vazia para armazenar as palavras corrigidas.
    novas_palavras = []

    # Todas as letras do alfabeto e as vogais acentuadas.
    letras = 'abcedfghijklmnopqrstuvwxyzáâàãéêèíîìóôòõúûùç'

    # Iterando por todas as tuplas da lista recebida
    for esquerdo, direito in palavra_fatiada:
        
        # Iterando por toda letra das variável letras
        for letra in letras:

            # Acrescentando todas as possibilidades de palavras possíveis
            novas_palavras.append(esquerdo + letra + direito)

    # Retornando uma lista de possíveis palavras
    return novas_palavras

### Função deleta_caracter.
def deleta_caracter(fatias):
    """
    Função para deletar um caracter depois que recebe as fatias.
    """

    # Criando uma lista vazia para armazenar as palavras corrigidas.
    novas_palavras = []

    # Iterando por todas as tuplas da lista recebida.
    for esquerdo, direito in fatias:

        # Acrescentando todas as possibilidades de palavras possíveis.
        novas_palavras.append(esquerdo + direito[1:])

    # Retornando uma lista de possíveis palavras.
    return novas_palavras

### Função troca_caracter.
def troca_caracter(fatias):
    """
    Função que recebe uma lista de tuplas (esquerdo, direito) que corresponde aos lados esquerdo e direito 
    da palavra fatiada em dois.
    """

    # Criando uma lista vazia para armazenar as palavras corrigidas.
    novas_palavras = []

    # Todas as letras do alfabeto e as vogais acentuadas.
    letras = 'abcedfghijklmnopqrstuvwxyzáâàãéêèíîìóôòõúûùç'

    # Iterando por todas as tuplas da lista recebida.
    for esquerdo, direito in fatias:

        # Iterando por toda letra das variável letras.
        for letra in letras:

            # Acrescentando todas as possibilidades de palavras possíveis.
            novas_palavras.append(esquerdo + letra + direito[1:])

    # Retornando uma lista de possíveis palavras.
    return novas_palavras

### Função inverte_caracter.
def inverte_caracter(fatias):
    """
    Função que recebe as fatias e inverte os caracteres.
    """

    # Criando uma lista vazia para armazenar as palavras corrigidas.
    novas_palavras = []

    # Iterando por todas as tuplas da lista recebida.
    for esquerdo, direito in fatias:
        
        # Selecionando apenas as fatias da direita que têm mais de uma letra, pois, se não, não há o que inverter.
        if len(direito) > 1:
            
            # Acrescentando todas as possibilidades de palavras possíveis.
            novas_palavras.append(esquerdo + direito[1] + direito[0] + direito[2:])

    # Retornando uma lista de possíveis palavras.
    return novas_palavras

### Função gera_palavras.
def gera_palavras(palavra):
    """
    Função para gerar possíveis palavras de acordo com a palavra passada e fatiada pela lógica.
    """

    # Lista vazia para armazenar as duas fatias de cada palavra.
    fatias = []

    # Iterando por cada letra de cada palavra.
    for i in range(len(palavra) + 1):

        # Armazenando as duas fatias em uma tupla e essa tupla em uma lista.
        fatias.append((palavra[:i], palavra[i:]))

    # Chamando a função 'insere_letras_faltantes' com a lista de tuplas das fatias recém-criadas.
    palavras_geradas = insere_letras_faltantes(fatias)

    # Acrescentando mais uma função, aqui que o novo corretor começa.
    palavras_geradas += deleta_caracter(fatias)

    # Acrescentando mais uma função, aqui que o novo corretor começa.
    palavras_geradas += troca_caracter(fatias)

    # Acrescentando mais uma função, aqui que o novo corretor começa.
    palavras_geradas += inverte_caracter(fatias)

    # Retornando a lista de possíveis palavras. A palavra correta estará aí no meio
    return palavras_geradas

### Função corretor_ortografico.
def corretor_ortografico(palavra_errada):
    """
    Função que recebe uma palavra errada, e retorna a palavra corrigida.
    """
    # Chama a função 'gera_palavras'.
    palavras_geradas = gera_palavras(palavra_errada)

    # Selecionando a palavra com maior probabilidade de aparecer em nosso corpus
    # Essa será a palavra correta
    palavra_correta = max(palavras_geradas, key=probabilidade_palavra)

    # Retornando a palavra corrigida
    return palavra_correta




# if st.checkbox("Mostrar o dataset"):
#     st.write(conteudo_treino[:500])

palavra_verificar = st.text_input('Digite aqui a sua palavra incorreta:')
if palavra_verificar:
    palavra_correta = corretor_ortografico(palavra_verificar)

if st.button('Verificar! ♻'):
    st.write("Resposta do corretor [", palavra_correta, "]")
