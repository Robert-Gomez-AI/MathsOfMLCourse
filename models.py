import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-dark')

np.random.seed(42)

def train_validation_test_split(X, y, test_size=0.2, val_size=0.25, random_state=42):

     # Dividir dataset en conjunto de entrenamiento + validación y conjunto de test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Dividir conjunto de entrenamiento + validación en conjunto de entrenamiento y conjunto de validación
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test



class SVM:

    def __init__(self, C = 1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0

   

     # Hinge Loss Function / Calculation
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]
    
    def accuracy(self,w,b,x,y):
        prediction = np.dot(x, w[0]) + b # w.x + b
        compared=np.sign(prediction)==y
        self.precision=sum(compared)/x.shape[0]
        return self.precision

    def fit(self, X, Y,X_val,y_val, batch_size=100, learning_rate=0.001, epochs=1000,eval_step=10):
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []
        val_loss=[]

        accuracy=[]
        val_accuracy=[]

        # Gradient Descent logic
        for i in range(epochs):
            if i%eval_step==0:
                # Calculating the Hinge Loss
                l = self.hingeloss(w, b, X, Y)
                vl= self.hingeloss(w,b,X_val,y_val)
                # Appending all losses 
                losses.append(l)
                val_loss.append(vl)

                acc= self.accuracy(w,b,X,Y)
                acc_val= self.accuracy(w,b,X_val,y_val)
                
                # Appending all losses 
                accuracy.append(acc)
                val_accuracy.append(acc_val)



            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial+ batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculating the gradients

                            #w.r.t w 
                            gradw += c * Y[x] * X[x]
                            # w.r.t b
                            gradb += c * Y[x]

                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b
        self.losses = losses
        self.val_loss= val_loss

        self.accuracy= accuracy
        self.val_accuracy= val_accuracy
        print(f'Precisión:{accuracy[-1]}')
        return self.w, self.b, losses

    def predict(self, X):
        
        prediction = np.dot(X, self.w[0]) + self.b # w.x + b
        return np.sign(prediction)
    def plot_loss(self):
        plt.plot(self.losses,linewidth=3,label='Losses')
        plt.plot(self.val_loss, label='val_loss')
        plt.suptitle('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(color='white')
        plt.legend()
        plt.show()
    
    def plot_omega(self):

        fig, axs = plt.subplots(1,3)
        axs[0].plot(self.losses,linewidth=3,label='Losses')
        axs[0].plot(self.val_loss, label='val_loss')
        axs[0].set_title('Loss vs Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        plt.grid(color='white')

        omega = np.abs(np.array(self.losses)-np.array(self.val_loss))

        axs[1].plot(omega,linewidth=3,label='Omega')
    
        axs[1].set_title('$\Omega$ vs Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('$\Omega$')
        plt.grid(color='white')

        axs[2].plot(self.accuracy,linewidth=3,label='Accuracy')
        axs[2].plot(self.val_accuracy, label='Validation accuracy')
        axs[2].set_title('Accuracy vs Epochs')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Accuracy')
        axs[2].legend()

        fig.tight_layout(pad=0.8)
        fig.set_figwidth(20)
        fig.set_figheight(5)
        plt.grid(color='white')
        plt.show()

    def evaluate_model(self,x, y,show_metrics=True):
        # Hacer la predicción con el modelo
        y_pred = self.predict(x)
        
        # Calcular los valores de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y[i] == 0 and y_pred[i] == 0:
                tn += 1
            elif y[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y[i] == 1 and y_pred[i] == 0:
                fn += 1
        print(f"True positives:{tp}")
        print(f"True negatives:{tn}")
        print(f"False positives:{fp}")
        print(f"False negatives:{tp}")

        # Calcular la precisión, el recall y el F1-score
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fp)
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        # Imprimir los resultados
        print("Precisión:", self.precision)
        print("Recall:", self.recall)
        print("F1-score:", self.f1_score) 
        self.confusion_matrix=np.array([[tn,fp],[fn,tp]])


    def plot_confusion_matrix(self):
        # Configuramos la figura y el subplot
        fig, ax = plt.subplots(figsize=(4, 4))
        confusion_matrix=self.confusion_matrix
        # Configuramos el mapa de colores y la escala
        im = ax.imshow(confusion_matrix, cmap='Blues')
        cbar = ax.figure.colorbar(im, ax=ax)

        # Añadimos los títulos y etiquetas de los ejes
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'],
            xlabel='Valor Predicho', ylabel='Valor Real')

        # Rotamos los labels de los ejes para que se lean mejor
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Añadimos los valores de cada celda de la matriz de confusión
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")

        # Ajustamos los márgenes de la figura
        fig.tight_layout()

        # Mostramos el plot
        plt.show()
