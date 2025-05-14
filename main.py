from app import PricePredictor

if __name__ == '__main__':
    predictor = PricePredictor('models/xgb_model.pkl')
    predictor.run()
