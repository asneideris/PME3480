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
#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 2. PARÂMETROS E CONSTANTES DO PROJETO Ana)
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
Vu = (np.pi/4.0)*(B**2)*S #Volume util do cilindro
omega = 2.0*np.pi*n    # rad/s
phi = 0.7 #razão de equivalência
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
