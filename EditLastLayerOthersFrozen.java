package org.deeplearning4j.examples.transferlearning.vgg16;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIterator;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

/**
 * @author susaneraly on 3/9/17.
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16
 * We will hold all layers but the very last one frozen and change the number of outputs in the last layer to
 * match our classification task.
 * In other words we go from where fc2 and predictions are vertex names in org.deeplearning4j.transferlearning.vgg16
 *  fc2 -> predictions (1000 classes)
 *  to
 *  fc2 -> predictions (5 classes)
 * The class "FitFromFeaturized" attempts to train this same architecture the difference being the outputs from the last
 * frozen layer is presaved and the fit is carried out on this featurized dataset.
 * When running multiple epochs this can save on computation time.
 */
public class EditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);

    protected static final int numClasses = 2;    //was 5
    protected static final long seed = 12345;

    private static int height = 224;
    private static int width = 224;
    private static int channels = 3;// here is a question///////////
    private static final int trainPerc = 80;
    private static final int valTrainPerc = 20;
    private static  int batchSize = 15;
    private static  int epochs = 5;
    private static String MAMMO_TRAIN_PATH = " " , MAMMO_TEST_PATH = " ";
    private static final String featureExtractionLayer = "fc2"; //was fc1;

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        String modelPath = " ";
        if (args.length < 3 || args.length > 4 ){
            System.out.println("Usage: program epochs pathToTrainData pathToTestModel pathToModel");
            System.exit(0);

        } else {
            epochs = Integer.parseInt(args[0]);
            MAMMO_TRAIN_PATH = args[1];
            MAMMO_TEST_PATH = args[2];
            modelPath = args[3];
        }
        log.info("pathToTrainData is: " + MAMMO_TRAIN_PATH);
        log.info("pathToTestModel is: " + MAMMO_TEST_PATH);
        log.info("modelPath is: " + modelPath);
        log.info("epochs are: " + epochs);

        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Nesterovs(5e-5))
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .dist(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numClasses)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "fc2")  //was fc2
            .build();
        log.info(vgg16Transfer.summary());

        //Dataset iterators
        FlowerDataSetIterator.setupTrainTest(batchSize, MAMMO_TRAIN_PATH , MAMMO_TEST_PATH, valTrainPerc);

          ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // In order to create a balanced dataset, the minimun of the two labels is selected



        // 2. select a subset of images for test (and eventually for validation)
        // (see into the source coude this comment)

        InputSplit trainData = FlowerDataSetIterator.trainData;
        InputSplit testData = FlowerDataSetIterator.testData;
        InputSplit valData = FlowerDataSetIterator.valData;

        boolean save = true;
        double maxAUC = 0;
        double AUC = 0;


        log.info("Train model....");

        for(int i = 0; i < epochs; i++){
            log.info("\n========================= Current epochs: " + (i + 1) + " ====================================");

            trainData.reset();
            ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
            recordReaderTrain.initialize(trainData, null);
            DataSetIterator dataIterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numClasses);
            dataIterTrain.reset();

            ImageRecordReader recordReaderVal = new ImageRecordReader(height, width, channels, labelMaker);
            recordReaderVal.initialize(valData);
            DataSetIterator dataIterVal = new RecordReaderDataSetIterator(recordReaderVal, batchSize, 1, numClasses);


            int iter = 0;
            // Train without transformations
            while(dataIterTrain.hasNext()) {

                if (iter % 10 == 0){
                    log.info("Evaluate model on validation at iter " + (iter + 1) + " ....");
                    maxAUC = evaluateModel(save, modelPath, vgg16Transfer, dataIterVal, maxAUC);
                }

                log.info("Train iter: " + (iter + 1));
                vgg16Transfer.fit(((RecordReaderDataSetIterator) dataIterTrain).next());

                iter++;
            }

        }

        log.info("Evaluate model on test set ....");

        //recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
        recordReaderTest.initialize(testData);
        DataSetIterator dataIterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numClasses);

        Evaluation eval = new Evaluation();
        ROC roc = new ROC();
        //Evaluation eval = vgg16Transfer.evaluate(dataIterVal);
        vgg16Transfer.doEvaluation(dataIterTest, eval, roc);

        log.info(eval.stats());

        // Evaluate AUC
        log.info("AUC on test: " + roc.calculateAUC());
        // Evaluate AUC under Precision Recall Curve
        log.info("AUCPR on test: " + roc.calculateAUCPR());


        log.info("****************Example finished********************");

    }

    private static double evaluateModel(boolean save, String modelPath, ComputationGraph vgg16Transfer, DataSetIterator dataIterVal, double maxAcc) throws IOException {


        Evaluation eval = new Evaluation();
        ROC roc = new ROC();
        double AUC = 0.0;
        double accuracy = 0.0;
        //Evaluation eval = vgg16Transfer.evaluate(dataIterVal);
        vgg16Transfer.doEvaluation(dataIterVal, eval, roc);
        accuracy = eval.accuracy();
        log.info(eval.stats());

        // Evaluate AUC
        AUC = roc.calculateAUC();
        
        log.info("AUC: " + AUC);
        // Evaluate AUC under Precision Recall Curve
        log.info("AUCPR: " + roc.calculateAUCPR());
        dataIterVal.reset();

        dataIterVal.reset();
        if (accuracy > maxAcc){
            maxAcc = accuracy;
            saveModel(save, modelPath, vgg16Transfer);
        }
        return maxAcc;
    }


    private static void saveModel(boolean save, String modelPath, ComputationGraph vgg16Transfer) throws IOException{
        if (save) {
            log.info("Save model to: " + modelPath);
            ModelSerializer.writeModel(vgg16Transfer, modelPath, true);
        }
    }
}
