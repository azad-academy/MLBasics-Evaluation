'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================

from linear_model import *
from nonlinear_model import *
from copy import deepcopy
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d


class Evaluation:

    def __init__(self,model,X_CV,Y_CV):

        self.X_train = model.X
        if(model.Y.ndim<2):
            model.Y = model.Y[:,np.newaxis]
        self.Y_train = model.Y
        self.X_CV = X_CV
        if(Y_CV.ndim<2):
            Y_CV = Y_CV[:,np.newaxis]
        self.Y_CV = Y_CV
        self.model = model
        self.CV_model = deepcopy(model)
        self.CV_model.X = X_CV
        self.CV_model.Y = Y_CV
        self.Jt = []
        self.Jcv = []

    def learning_curves(self, L=1):
        
        fig,ax = plt.subplots(1,1,figsize=(10,8))

        
        m_train = len(self.model.X)
        m_cv = len(self.CV_model.X)
        m = min([m_train,m_cv])
        x = range(int(m/12),m,int(m/12))
        for i in x: 
            indices_train = np.random.randint(m_train,size=i)
            indices_cv = np.random.randint(m_cv,size=i)
            self.model.X = self.X_train[indices_train,:]
            self.model.Y = self.Y_train[indices_train]
            self.CV_model.X = self.X_CV[indices_cv,:]
            self.CV_model.Y = self.Y_CV[indices_cv]
            with np.errstate(divide='ignore', invalid='ignore'):
                P = self.model.train(100,L)
                J = self.Jt[-1] if math.isnan(P.fun) else P.fun   
                self.Jt.append(J)
                P = self.CV_model.train(100,L)
                J = self.Jcv[-1] if math.isnan(P.fun) else P.fun
                self.Jcv.append(J)

        X_ = np.linspace(min(list(x)), max(list(x)), 500)
        Jt_curve = make_interp_spline(x, self.Jt)
        Jcv_curve = make_interp_spline(x, self.Jcv)
        Jt_line, = ax.plot(X_,Jt_curve(X_),'r-',linewidth=2)
        Jcv_line, = ax.plot(X_,Jcv_curve(X_),'b-',linewidth=2)
        ax.set_xlabel('Samples(m)')
        ax.set_ylabel('Objective Function (J)')
        Jt_line.set_label('$J_{{train}}$')
        Jcv_line.set_label('$J_{{CV}}$')
        ax.legend()
        limit = list(ax.get_ylim())
        limit[0] = 0
        ax.set_ylim(limit)
        return fig,ax


        

    def compute_performance(self,X,Y,thresh=0.5):
        
        Y_hat = self.model.predict(X,thresh)
        
        
        TP = sum(Y*Y_hat)
        TN = sum((1-Y)*(1-Y_hat))
        FP = sum((1-Y)*Y_hat)
        FN = sum(Y*(1-Y_hat))

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        FPR = FP/(FP+TN)

        return round(accuracy,2),round(precision,2),round(recall,2),round(FPR,2)


    def show_performance_curves(self):
        fig,axes = plt.subplots(1,2,figsize=(12,5))
        TPRS = []
        FPRS = []
        precisions = []
        TPRS_CV = []
        FPRS_CV = []
        precisions_CV = []
         
        for thresh in np.arange(0,1,0.01):
            accuracy, precision, TPR, FPR = self.compute_performance(self.X_train,self.Y_train.squeeze(),thresh)
            TPRS.append(TPR)
            FPRS.append(FPR)
            precisions.append(precision)
            accuracy, precision, TPR, FPR = self.compute_performance(self.X_CV,self.Y_CV.squeeze(),thresh)
            TPRS_CV.append(TPR)
            FPRS_CV.append(FPR)
            precisions_CV.append(precision)

        X_ = np.linspace(min(list(FPRS)), max(list(FPRS)), 500)
        curve = interp1d(FPRS, TPRS, fill_value="extrapolate")
        cv_curve = interp1d(FPRS_CV, TPRS_CV, fill_value="extrapolate")
        ax = axes[0]
        
        ax.fill_between(X_, curve(X_), step="pre", alpha=0.1)
        ax.fill_between(X_, cv_curve(X_), step="pre", alpha=0.1)
        ax.plot(X_,curve(X_),'r-',label='Learned Classifier(trainset)',linewidth=2)
        ax.plot(X_,cv_curve(X_),'b-',label='Learned Classifier(Cross Validation)',linewidth=2)
        ax.plot([0,1],[0,1],'k--',label='No-Skill Classifier',linewidth=2)
        ax.set_xlim([0,1.01])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall) ')
        ax.legend()


        X_ = np.linspace(min(list(TPRS)), max(list(TPRS)), 500)
        curve = interp1d(TPRS, precisions, fill_value="extrapolate")
        cv_curve = interp1d(TPRS_CV, precisions_CV, fill_value="extrapolate")
        Y = self.Y_train.squeeze()
        no_skill = len(Y[Y==1]) / len(Y)
        ax = axes[1]
        ax.plot(X_,curve(X_),'r-',label='Learned Classifier(trainset)',linewidth=2)
        ax.plot(X_,cv_curve(X_),'b-',label='Learned Classifier(Cross Validation)',linewidth=2)
        ax.plot([0,1],[no_skill,no_skill],'k--',label='No-Skill Classifier',linewidth=2)
        ax.set_xlim([0,1.01])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('True Positive Rate (Recall)')
        ax.set_ylabel('Precision (PPV)')
        ax.legend()
        
        return fig,ax



    
