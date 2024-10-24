# main.py
from app import create_app
import config
from services.data_integration import Constellations
from services.gnss import Satellite

app = create_app()
app.config.from_object(config.DevelopmentConfig)

if __name__ == '__main__':
    # integrate available constellations
    constellations = Constellations()
    constellations.update_sats()


    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])