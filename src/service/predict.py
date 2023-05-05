from src.service.train_model import use_model

def predict_text(text):
    model = use_model()
    prediction = model.predict([text])

    if prediction[0] > 0.5:
        #print('Positivo')
        
        """
        return {
            'classification': 'Positivo',
            'score': float(prediction[0])
        }
        """
    else:
        
        #print('Negativo')
        
        """
        return {
            'classification': 'Negativo',
            'score': float(prediction[0])
        }
        """

predict_text('No me llamen más, no quiero saber nada de ustedes')