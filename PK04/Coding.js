// Center the map and set options
Map.centerObject(roi, 13);
Map.setOptions('satellite');

// Load Sentinel-2 image and preprocess
var preImage = s2.filterBounds(roi)
                 .filterDate('2024-01-01', '2024-11-30')
                 .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10);
//s
var image = preImage.map(function(image) {
  return image.multiply(0.0001);
}).median().clip(roi);

Map.addLayer(image, {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.3
}, 'True Color Composite');

// Define training regions and labels
var training = ee.FeatureCollection([KerangasKolam, KerangasRawa, KerangasHutan]).flatten();
var predictionBands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12'];

var labeledTraining = image.select(predictionBands)
                           .sampleRegions({
                             collection: training,
                             properties: ['class'], // Ensure 'class' column exists
                             scale: 10
                           });

var split = labeledTraining.randomColumn();
var trainingData = split.filter(ee.Filter.lt('random', 0.8));
var testingData = split.filter(ee.Filter.gte('random', 0.8));

// Generalized function to train, classify, and evaluate models
function trainAndClassify(classifier, name, palette) {
  try {
    var trainedClassifier = classifier.train({
      features: trainingData,
      classProperty: 'class',
      inputProperties: predictionBands
    });

    var classified = image.select(predictionBands).classify(trainedClassifier);

    Map.addLayer(classified, {
      min: 0,
      max: 2, // Adjust based on the number of classes
      palette: palette
    }, name + ' Classified Map');

    // Validation
    var validation = testingData.classify(trainedClassifier);
    var errorMatrix = validation.errorMatrix('class', 'classification');
    print(name + ' Confusion Matrix:', errorMatrix);
    print(name + ' Overall Accuracy:', errorMatrix.accuracy());
  } catch (error) {
    print(name + ' Error:', error.message);
  }
}

// Random Forest Classifier
trainAndClassify(ee.Classifier.smileRandomForest(100), 'Random Forest', ['red', 'green', 'blue']);

// Support Vector Machine (SVM)
trainAndClassify(ee.Classifier.libsvm(), 'SVM', ['yellow', 'magenta', 'cyan']);

// Gradient Tree Boost Classifier
trainAndClassify(ee.Classifier.smileGradientTreeBoost(50), 'Gradient Boost', ['orange', 'purple', 'lime']);

// CART Classifier
trainAndClassify(ee.Classifier.smileCart(100, 5), 'CART', ['lightblue', 'darkgreen', 'yellow']);

// K-Nearest Neighbors (KNN) (FIXED)
trainAndClassify(ee.Classifier.smileKNN(5, 'AUTO', 'EUCLIDEAN'), 'KNN', ['teal', 'maroon', 'gold']);

// Define function for K-Fold Cross Validation
function kFoldCrossValidation(k, classifier, data, predictionBands, palette) {
  var results = [];
  var kappaScores = [];
  
  // Add fold number as a property to the dataset
  var dataWithFolds = data.randomColumn('fold').map(function(feature) {
    var foldNumber = ee.Number(feature.get('fold')).multiply(k).floor();
    return feature.set('foldNumber', foldNumber);
  });
  
  for (var i = 0; i < k; i++) {
    // Split data into training and testing for this fold
    var trainingFold = dataWithFolds.filter(ee.Filter.neq('foldNumber', i));
    var testingFold = dataWithFolds.filter(ee.Filter.eq('foldNumber', i));
    
    // Train classifier
    var trainedClassifier = classifier.train({
      features: trainingFold,
      classProperty: 'class',
      inputProperties: predictionBands
    });
    
    // Validate
    var validation = testingFold.classify(trainedClassifier);
    var errorMatrix = validation.errorMatrix('class', 'classification');
    var accuracy = errorMatrix.accuracy();
    var kappa = errorMatrix.kappa();

    // Store metrics
    results.push(accuracy);
    kappaScores.push(kappa);
  }

  // Calculate averages
  var meanAccuracy = ee.Array(results).reduce(ee.Reducer.mean(), [0]).get([0]);
  var meanKappa = ee.Array(kappaScores).reduce(ee.Reducer.mean(), [0]).get([0]);
  
  print('K-Fold Mean Accuracy:', meanAccuracy);
  print('K-Fold Mean Kappa:', meanKappa);

  return {
    accuracy: meanAccuracy,
    kappa: meanKappa
  };
}

// Apply K-Fold Cross Validation for each classifier
var k = 5; // Number of folds
kFoldCrossValidation(k, ee.Classifier.smileRandomForest(100), labeledTraining, predictionBands, ['red', 'green', 'blue']);
kFoldCrossValidation(k, ee.Classifier.libsvm(), labeledTraining, predictionBands, ['yellow', 'magenta', 'cyan']);
kFoldCrossValidation(k, ee.Classifier.smileGradientTreeBoost(50), labeledTraining, predictionBands, ['orange', 'purple', 'lime']);
kFoldCrossValidation(k, ee.Classifier.smileCart(100, 5), labeledTraining, predictionBands, ['lightblue', 'darkgreen', 'yellow']);
kFoldCrossValidation(k, ee.Classifier.smileKNN(5, 'AUTO', 'EUCLIDEAN'), labeledTraining, predictionBands, ['teal', 'maroon', 'gold']);

// Export training data
Export.table.toDrive({
  collection: labeledTraining,
  description: 'TrainingDataExport',
  fileFormat: 'CSV'
});
