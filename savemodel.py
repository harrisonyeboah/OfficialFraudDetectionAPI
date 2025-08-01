from model import train_model
import joblib

rf_model, scaler = train_model()
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


