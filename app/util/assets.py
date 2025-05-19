from flask_assets import Bundle, Environment
from .. import app

bundles = {
    'js': Bundle(  
        'js/libs/d3.js',
        'js/libs/jquery-3.4.1.min.js',
        'js/libs/leader-line.min.js',
        'js/libs/bootstrap.min.js',
        'js/assignment/inputFairMetric.js',
        'js/confusionMtxPanel.js',
        'js/VCSetHandler.js',
        'js/init.js',
        'js/encryptWeb.js',
        'js/renderVCHandler.js',
        'js/trainTestPanel.js',
        'js/accuracyPanel.js',
        'js/LRExplainerPanel.js',
        'js/preProcessPanel.js',
        'js/postProcessPanel.js',
        'js/latexHandler.js',
        'js/global.js',
        'js/index.js',
        'js/fairMetrics.js',
        'js/customization.js',
        'js/assignment/fairMetricsFPanel.js',
        'js/assignment/TrainModelCusPanel.js',
        'js/assignment/inputFPanel.js',
        output='gen/script.js'
        ),

        'css': Bundle(
        'css/bootstrap/bootstrap.css',
        'css/bootstrap/layout-bootstrap.css',
        'css/layout.css',
        'css/content.css',
        'css/assignment/inputFPanel.css',
        output='gen/styles.css'
        )
}

assets = Environment(app)

assets.register(bundles)