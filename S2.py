#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

# PME-3480 - Motores de Combustão Interna
# 1D Otto cycle simulator - 2025
# Implementation 2 - Group 01

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
#-----------------------------------------------------------------------------#
import numpy as np
import OttoCycle as oc
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cantera as ct
import os

ct.add_directory(os.getcwd())
#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 2. PARÂMETROS E CONSTANTES DO PROJETO (Ana)
#-----------------------------------------------------------------------------#
# Parâmetros fixos do motor (Grupo 1)
B = 60 / 1000      # Diâmetro do cilindro (m)
S = 120 / 1000     # Curso do pistão (m)
L = 180 / 1000     # Comprimento da biela (m)
n_rpm = 2000       # Rotação do motor (rpm)
n = n_rpm / 60     # Rotação do motor (rps)
fuel = 'CH4'       # Combustível (Metano)
Zc = 1             # nº de cilindros (S1)
x  = 2             # motor 4T -> 2 voltas por ciclo
Vu = (np.pi/4.0)*(B**2)*S # Volume util do cilindro
omega = 2.0*np.pi*n    # rad/s
phi = 0.7 # razão de equivalência
AFest = (4.76*2*28.97)/16.043 # razão ar-comb estequiométrica
AFreal = AFest/phi # razão ar-comb real
rho_ar = 1.184 # massa específica de ar na condição de teste
#rho_ar = 1.1694 # densidade do ar em 25°C e 1 atm [kg/m^3] - obtido pelo Laine

# Parâmetros de contorno
pint = 100e3       # Pressão de admissão (Pa)
Tint = 273.15 + 25 # Temperatura de admissão (K)
pexh = 100e3       # Pressão de escape (Pa)
Texh = 515 +273.15 # Temperatura de exaustão (K) (Obtida no S1 para rv = 10)
rv = 10            # Taxa de compressão

# Constantes físicas
PCI_CH4 = 50.01e6  # Poder Calorífico Inferior do Metano (J/kg)

# Ângulo do virabrequim para a simulação
#-----------------------------------------------------------------------------#
# Admissão
#-----------------------------------------------------------------------------#
ThIVO = +360. * (np.pi / 180.)  # Abertura da válvula de admissão (convertida para radianos)
ThIVC = -150. * (np.pi / 180.)  # Fechamento da válvula de admissão (convertida para radianos)

#-----------------------------------------------------------------------------#
# Exaustão
#-----------------------------------------------------------------------------#
ThEVO = +150. * (np.pi / 180)  # Abertura da válvula de exautão (convertida para radianos)
ThEVC = -360. * (np.pi / 180)  # Fechamento da válvula de exaustão (convertida para radianos))

#-----------------------------------------------------------------------------#
# 3. LEITURA DOS DADOS EXPERIMENTAIS (Ana)
#-----------------------------------------------------------------------------#
dados_exp = np.loadtxt('grupo01-massFractionBurned.txt', skiprows=5, encoding='latin1')
cad = dados_exp[:, 0]  # CAD (first column)
xb1 = dados_exp[:, 1]  # xb1 (second column)
xb2 = dados_exp[:, 2]  # xb2 (third column)
xb3 = dados_exp[:, 3]  # xb3 (fourth column)

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 4. LOOP PRINCIPAL DE SIMULAÇÃO E CÁLCULOS
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 5. AJUSTE DA FUNÇÃO DE WIEBE (Matheus)
#-----------------------------------------------------------------------------#

# Definição da Função de Wiebe teórica para o ajuste
# x = theta (CAD), a = eficiência, m = forma, th_soc = início, duration = duração
def wiebe_model(theta, a, m, th_soc, duration):
    # Previne divisão por zero ou potências inválidas
    theta_norm = (theta - th_soc) / duration

    # A função deve retornar 0 antes do SOC e 1 após o fim (SOC+duration)
    # Usamos np.where para vetorizar essa lógica condicional
    val = 1.0 - np.exp(-a * (theta_norm ** (m + 1.0)))

    val = np.where(theta < th_soc, 0.0, val) # Antes da combustão é 0
    val = np.where(theta > (th_soc + duration), 1.0, val) # Depois é 1
    return val

# Lista para armazenar os parâmetros ajustados dos 3 casos
# Formato: [ [a1, m1, SOC1, EOC1], [a2, m2, SOC2, EOC2], ... ]
wiebe_params_list = []

# Loop para ajustar as 3 curvas (xb1, xb2, xb3)
# Note que i+1 é usado apenas para printar "Caso 1, 2, 3"
curvas_xb = [xb1, xb2, xb3]

print("\n--- INICIANDO AJUSTE DE PARÂMETROS WIEBE ---")

for i, xb_exp in enumerate(curvas_xb):
    # 1. Filtragem de dados: O otimizador funciona melhor se pegarmos apenas
    # a região onde a combustão está realmente ocorrendo (ex: 0.1% a 99.9%)
    # Isso evita que ele se perca nos infinitos zeros ou uns.
    mask = (xb_exp > 0.001) & (xb_exp < 0.999)

    cad_fit = cad[mask]
    xb_fit  = xb_exp[mask]

    # 2. Chute inicial (Initial Guess) - Ajuda o solver a não convergir para algo absurdo
    # a=5, m=2, soc=-10, duration=40 é um chute padrão razoável
    p0 = [5.0, 2.0, -15.0, 50.0]

    # 3. Limites (Bounds) - [minimos], [maximos]
    # a>0, m>0, soc entre -50 e 0, duração entre 10 e 100
    bounds = ([0.1, 0.1, -60.0, 10.0], [20.0, 10.0, 10.0, 120.0])

    # 4. Otimização
    try:
        popt, pcov = curve_fit(wiebe_model, cad_fit, xb_fit, p0=p0, bounds=bounds)

        a_opt, m_opt, soc_opt, dur_opt = popt
        eoc_opt = soc_opt + dur_opt

        # Armazena para o Felipe usar depois
        wiebe_params_list.append({
            'a': a_opt,
            'm': m_opt,
            'SOC_deg': soc_opt,
            'EOC_deg': eoc_opt,
            'duration': dur_opt
        })

        print(f"--> Caso {i+1} ajustado:")
        print(f"    a = {a_opt:.4f}")
        print(f"    m (input) = {m_opt:.4f}  [OBS: No simulador usa-se mWF = m_input]")
        print(f"    SOC = {soc_opt:.2f}° | EOC = {eoc_opt:.2f}° | Duração = {dur_opt:.2f}°")

        # Comentado: Plot rápido de validação/prova (apenas se rodar local, não trava o script)
        # plt.plot(cad, xb_exp, 'k.', label='Exp')
        # plt.plot(cad, wiebe_model(cad, *popt), 'r-', label='Fit')
        # plt.show()

    except RuntimeError:
        print(f"ERRO: Não foi possível ajustar a curva {i+1}.")

print("---------------------------------------------------\n")

#-----------------------------------------------------------------------------#
# 6. SIMULAÇÃO DO CICLO COM PARÂMETROS AJUSTADOS (Felipe)
#-----------------------------------------------------------------------------#

print("--- INICIANDO SIMULAÇÕES DO CICLO OTTO (S2) ---")

# 6.1. Definição do vetor de ângulos para a simulação (Domínio do Tempo)
# Faz-se necessário rodar o ciclo completo (-360 a 360), independente dos dados experimentais
Th0 = -360. * (np.pi/180)
Th1 = +360. * (np.pi/180)
Ths = 1e-1 * (np.pi/180)  # Passo de 0.1 grau para alta resolução
Thn = int(((Th1 - Th0) / Ths) + 1)
Th = np.linspace(start=Th0, stop=Th1, num=Thn, endpoint=True)
CAD_sim = Th * (180 / np.pi)

# Dicionário para armazenar todos os resultados
dados_simulacao = {
    'CAD': CAD_sim, # Eixo X comum a todos
    'casos': []     # Lista para guardar os 3 dicionários de resultados
}

# Loop principal: Roda o OttoCycle para cada conjunto de parâmetros Wiebe descobertos
for i, params in enumerate(wiebe_params_list):
    print(f"--> Simulando Caso {i+1}...")
    
    # 6.2. Conversão de unidades
    # OttoCycle requer em radianos
    a_fit = params['a']
    m_fit = params['m']
    SOC_rad = params['SOC_deg'] * (np.pi / 180.0)
    EOC_rad = params['EOC_deg'] * (np.pi / 180.0)
    
    # 6.3. Montagem da tupla 'pars'
    # rv=10 e Texh foram definidos no início
    pars = (
        'fired',       # case
        B, S, L,       # Geometria
        rv,            # Taxa de compressão (fixa em 10)
        n,             # Rotação
        ThIVO, ThIVC,  # Válvulas Admissão
        ThEVO, ThEVC,  # Válvulas Escape
        SOC_rad,       # Start of Combustion
        EOC_rad,       # End of Combustion
        a_fit,         # Wiebe a
        m_fit,         # Wiebe m
        pint, Tint,    # Admissão
        pexh, Texh,    # Exaustão
        phi, fuel      # Mistura
    )
    
    # 6.4. Execução do Solver (Cálculo de Pressão e Temperatura)
    # oc.ottoCycle retorna: Volume, Massa, Temperatura, Pressão
    V_sim, m_sim, T_sim, p_sim = oc.ottoCycle(Th, pars)
    
    # 6.5. Cálculo da Fração Queimada (xb)
    # Faz-se necessário chamar essa função explicitamente pois o ottoCycle não devolve o xb
    xb_sim, dxb_sim = oc.wiebeFunction(Th, pars)
    
    # Armazena no dicionário do caso
    caso_dict = {
        'id': i+1,
        'p': p_sim,       # Pressão [Pa]
        'T': T_sim,       # Temperatura [K]
        'xb': xb_sim,     # Fração queimada [-]
        'dxb': dxb_sim,   # Taxa de queima
        'pars': pars,     # Salva os parâmetros usados para referência
        'wiebe_data': params # Salva os dados brutos do ajuste (a, m, soc, eoc em graus)
    }
    
    dados_simulacao['casos'].append(caso_dict)

#-----------------------------------------------------------------------------#
# 7. SALVAMENTO DOS DADOS
#-----------------------------------------------------------------------------#
# O guia pede para salvar na pasta Files em formato .npy
nome_arquivo = './Files/S2_Resultados_Grupo01.npy'
np.save(nome_arquivo, dados_simulacao)

print("\n" + "="*50)
print(f"✅ TAREFA CONCLUÍDA (Felipe)")
print(f"Arquivo salvo: {nome_arquivo}")
print("Conteúdo: Dicionário contendo CAD e uma lista com P, T, xb, dxb para os 3 casos.")
print("Carregar este arquivo usando: dados = np.load('...', allow_pickle=True).item()")
print("="*50)
