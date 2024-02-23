import numpy as np
from scipy.stats import f
from scipy.stats import ttest_ind
import scipy.stats as stats


def mad_2(datos_4):
  """
  Calcula la desviación absoluta media (MAD) de una serie de datos.
  """
  mean = datos_4.mean()
  deviations = abs(datos_4 - mean)
  return deviations.mean()


def var_test(x, y, ratio=1, alternative="two-sided", conf_level=0.95):
  # Eliminar valores faltantes
  x = np.array(x)[np.isfinite(x)]
  y = np.array(y)[np.isfinite(y)]

  # Calcular grados de libertad y varianzas
  df_x = len(x) - 1
  df_y = len(y) - 1
  var_x = np.var(x, ddof=1)
  var_y = np.var(y, ddof=1)

  # Calcular estadístico F y p-value
  estimate = var_x / var_y
  statistic = estimate / ratio
  pval = f.sf(statistic, df_x, df_y)
  if alternative == "two-sided":
    pval = 2 * min(pval, 1 - pval)
    beta = (1 - conf_level) / 2
    cint = [estimate / f.ppf(1 - beta, df_x, df_y), estimate / f.ppf(beta, df_x, df_y)]
  elif alternative == "greater":
    pval = 1 - pval
    cint = [estimate / f.ppf(conf_level, df_x, df_y), np.inf]
  else:
    cint = [0, estimate / f.ppf(1 - conf_level, df_x, df_y)]

  # Crear objeto htest y devolver resultado
  htest = {
    "statistic": statistic,
    "parameter": {"num df": df_x, "denom df": df_y},
    "p.value": pval,
    "conf.int": cint,
    "estimate": estimate,
    "null.value": ratio,
    "alternative": alternative,
    "method": "F test to compare two variances",
    "data.name": f"{x} and {y}"
  }

  return htest['p.value']


def ttest_pvalue_scipy(d1, d2, equal_vari):
  """Recibe dos series de datos numpy y un booleano de si las varianzas son iguales.
   Devuelve el resultado del t-test de scipy.
   """
  a = ttest_ind(d1, d2, equal_var=equal_vari)
  return a[1]


def inv_t_test_b(alfa, free_deg):
  """recibe un valor flotante (alfa) y un valor entero (free_deg) y devuelve el t test inverso de forma b"""
  return stats.t.ppf((1 - (1 - alfa) / 2), free_deg)


def inv_t_test_a(alfa, free_deg):
  """recibe un valor flotante (alfa) y un valor entero (free_deg) y devuelve el t test inverso de forma a"""
  return stats.t.ppf((1 - alfa / 2), free_deg)


def promedio_serie(s1, prueba):
  if prueba:
    return s1.mean()
  return -1


def varianza_serie(s1, prueba, magnitud, alt):
  if prueba:
    if alt:
      return s1.var()
    if magnitud:
      return s1.var()
    if mad_2(s1) > 1:
      return s1.var()
  if alt:
    return -2
  return -1


def prueba_f_serie(s1, s2, var1, var2):  # -1 ="no calcular" -2="revisar"
  if var1 == -1:
    return -1
  if (var2 == 0) or (var2 == -2):
    return -2
  return var_test(s1, s2)


def lib_grad(s1, s2, var1, var2, pruebaf):  # -1 = "no calcular " -2 = "revisar"
  if pruebaf == -2:
    return -2
  if (var1 == -1) or (var2 == -2):
    return -1
  if s1.size + s2.size - 2 < 0:
    return -1
  return s1.size + s2.size - 2


def prueba_t_serie(s1, s2, pruebaf, grados_lib):  # -2000000 = "revisar" -1000000 = "no calcular"
  if grados_lib == -2:
    return -2000000
  if grados_lib == -1:
    return -1000000
  if pruebaf < 0.05:
    equal_var = False
  else:
    equal_var = True
  # s1=
  return ttest_pvalue_scipy(s1, s2, equal_var)


def anti_t_student(t_st, glib, dug):
  if t_st == -2000000:
    return t_st
  if glib == -1:
    return -1000000
  if dug:
    return inv_t_test_a(t_st, glib)
  return inv_t_test_b(t_st, glib)
