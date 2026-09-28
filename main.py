import random


class Neuronio:
    # Cada neurónio tem um vetor de pesos w
    # Uma dimensão de entradas d
    # Um pendor b
    # Uma função de ativação sigma
    # Um h = x.w + b
    # Uma saída y
    # Temos também uma derivada y que representa a influência de algo sob o resultado
    def __init__(self, d, sigma):
        self.w = [random.uniform(-1, 1) for _ in range(d)]  # a list of random numbers between the interval in range d
        self.b = random.uniform(-1, 1)
        self.h = 0
        self.y = 0
        self.d = d
        self.sigma = sigma
        self.dy = 0

    # Função que faz o neurónio trabalhar
    def propagar(self, x):
        # Calcular h = x.w + b, isto é um produto escalar + o pendor
        self.h = sum(curr_w * curr_x for curr_w, curr_x in zip(self.w, x)) + self.b
        # Aqui tratamos a função de ativação como se fosse um objeto com o método ativar e o método derivar
        self.y = self.sigma.ativar(self.h)
        # A derivada de y passa a ser igual à influência da função de ativação no h
        self.dy = self.sigma.derivar(self.h)
        return self.y

    # alpha é um valor
    # propag_erro é um valor
    # dy é um valor
    # y_entrada é um array
    # delta_w é um array
    # Esta função tem como objetivo ajustar os valores dos pesos e do pendor de acordo com os resultados da iteração anterior
    def adaptar(self, erro, y_entrada, alpha):
        delta_w = [-alpha * self.dy * erro * y_i for y_i in y_entrada]
        self.w = [dw_i + w_i for dw_i, w_i in zip(delta_w, self.w)]
        delta_b = -alpha * self.dy * erro
        self.b = self.b + delta_b


# Primeira parte da rede neuronal, regista as entradas e propaga par a próxima camada
class CamadaEntrada:
    # Uma camada de entrada tem um y com tantas entradas quanto a dimensão de saída
    def __init__(self, ds):
        self.ds = ds
        self.y = [0 for _ in range(ds)]

    # Recebe entradas e entrega a próxima camada
    def propagar(self, x):
        self.y = x
        return self.y


# Segunda e terceira parte da rede neuronal
class CamadaDensa:
    # Cada camada densa/de saída tem uma dimensão de entrada(recebidas da camada anterior) e outra de saída(para enviar à próxima camada)
    # Tem n neurónios e uma função de ativação
    def __init__(self, de, ds, sigma):
        self.de = de
        self.ds = ds
        self.sigma = sigma
        self.neuronios = [Neuronio(de, sigma) for _ in range(ds)]

    # Propriedade y que representa as saídas de cada neurónio
    @property
    def y(self):
        return [neuronio.y for neuronio in self.neuronios]

    # Função usada para enviar as saídas à próxima camada
    def propagar(self, x):
        y = [neuronio.propagar(x) for neuronio in self.neuronios]
        return y

    # Chama a função adaptar de cada neurónio para ajustar os neurónios todos de acordo com os resultados anteriores
    def adaptar(self, erro, y_anterior, alpha):
        for j in range(0, self.ds): self.neuronios[j].adaptar(erro[j], y_anterior, alpha)


# Classe que une tudo
# É criada uma lista de camadas adicionando logo a camada de entrada, depois iteramos N vezes e criamos camadas escondidas/a de saída
class Rede_Neuronal:
    # Uma rede neuronal tem uma lista de camadas
    # Começamos por criar logo a camada de entrada
    # Criamos também N camadas a começar pela camda densa que é o item [1] da forma da rede
    # Para formar uma camada densa passamos forma[i-1] que representa a dimenão de saída da camada anterior e passamos forma[i] que representa a dimensão de saída
    def __init__(self, forma, sigma):
        self.camadas = []
        N = len(forma)
        ds1 = forma[0]
        camada1 = CamadaEntrada(ds1)
        self.camadas.append(camada1)
        for i in range(1, N):
            self.camadas.append(CamadaDensa(forma[i - 1], forma[i], sigma))

    # Função que faz com que a camada de entrada passe as suas entradas à camada densa e então a camada densa calcula o vetor y para a próxima camada e por aí adiante
    def propagar(self, x):
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y

    def prever(self, X):
        Y = [self.propagar(x) for x in X]
        return Y

    # Mostra a diferença entre os resultados reais e os resultados esperados
    def delta_saida(self, y, y_esperado):
        return [y_i - y_esperado_i for y_i, y_esperado_i in zip(y, y_esperado)]

    def retropropagar(self, erro, alpha):
        for i in range(len(self.N), 2, -1):
            y_anterior = self.camadas[i - 1].y
            d_anterior = self.camadas[i - 1].ds
            d_atual = self.camadas[i].ds
            neur_atual = self.camadas[i].neuronios
            erro_anterior = for j in range(1, len(d_atual)): sum(neur_atual[j].w)
