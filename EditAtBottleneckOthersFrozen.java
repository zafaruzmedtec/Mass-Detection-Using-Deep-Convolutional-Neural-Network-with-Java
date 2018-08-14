package org.deeplearning4j.examples.transferlearning.vgg16;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.MammogramDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * @author susaneraly on 3/1/17.
 *
 * IMPORTANT:
 * 1. The forward pass on VGG16 is time consuming. Refer to "FeaturizedPreSave" and "FitFromFeaturized" for how to use presaved datasets
 * 2. RAM at the very least 16G, set JVM mx heap space accordingly
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16.
 * We keep block5_pool and below frozen
 *      and modify/add dense layers to form
 *          block5_pool -> flatten -> fc1 -> fc2 -> fc3 -> newpredictions (5 classes)
 *       from
 *          block5_pool -> flatten -> fc1 -> fc2 -> predictions (1000 classes)
 *
 * Note that we could presave the output out block5_pool like we do in FeaturizedPreSave + FitFromFeaturized
 * Refer to those two classes for more detail
 */
public class EditAtBottleneckOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckOthersFrozen.class);

    protected static final int numClasses = 2;

    protected static final long seed = 12345;
    private static int height = 224;
    private static int width = 224;
    private static int channels = 3;
    private static final int trainPerc = 80;
    private static final int batchSize = 1;
    private static final String featureExtractionLayer = "block5_pool";

    private static final int valTrainPerc = 10;
    private static  int epochs = 2;
    private static String MAMMO_TRAIN_PATH = " " , MAMMO_TEST_PATH = " ";

    public static void main(String [] args) throws Exception {
        //DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
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
            .activation(Activation.TANH)//LEAKYRELU
            .weightInit(WeightInit.XAVIER)//RELU
            .updater(new Nesterovs(5e-5))
            .dropOut(0.5)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        //  Note: This architecture is constructed with the primary intent of demonstrating use of the transfer learning API,
        //        secondary to what might give better results
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //"block5_pool" and below are frozen
            .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
            .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
            .addLayer("fc3",new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
            .addLayer("newpredictions",new OutputLayer
                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(256)
                .nOut(numClasses)
                .build(),"fc3") //add in a final output dense layer,
            // note that learning related configurations applied on a new layer here will be honored
            // In other words - these will override the finetune confs.
            // For eg. activation function will be softmax not RELU
            .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
            .build();
        log.info(vgg16Transfer.summary());

        //Dataset iterators
        MammogramDataSetIterator.setupTrainTest(batchSize, MAMMO_TRAIN_PATH , MAMMO_TEST_PATH, valTrainPerc);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        InputSplit trainData = MammogramDataSetIterator.trainData;
        InputSplit testData = MammogramDataSetIterator.testData;
        InputSplit valData = MammogramDataSetIterator.valData;

        ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
        recordReaderTest.initialize(testData);
        DataSetIterator dataIterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numClasses);

        Evaluation eval;
        eval = vgg16Transfer.evaluate(dataIterTest);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        dataIterTest.reset();

        boolean save = true;
        double maxACC = 0;

        int iter = 0;
        for(int i = 0; i < epochs; i++)
        {
            log.info("\n========================= Current epochs: " + (i + 1) + " ====================================");

            trainData.reset();
            ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
            recordReaderTrain.initialize(trainData, null);
            DataSetIterator dataIterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numClasses);
            dataIterTrain.reset();

            ImageRecordReader recordReaderVal = new ImageRecordReader(height, width, channels, labelMaker);
            recordReaderVal.initialize(valData);
            DataSetIterator dataIterVal = new RecordReaderDataSetIterator(recordReaderVal, batchSize, 1, numClasses);


            iter = 0;
            // Train without transformations
            while (dataIterTrain.hasNext())
            {

                if (iter % 10 == 0) {
                    log.info("Evaluate model on validation at iter " + (iter + 1) + " ....");
                    maxACC = evaluateModel(save, modelPath, vgg16Transfer, dataIterVal, maxACC);
                }

                log.info("Train iter: " + (iter + 1));
                vgg16Transfer.fit(((RecordReaderDataSetIterator) dataIterTrain).next());

                iter++;
            }
        }

        log.info("Model build complete");
        // Evaluate the model on test set
            log.info("Evaluate model on test set After fitting ....");

            eval = vgg16Transfer.evaluate(dataIterTest);
            log.info(eval.stats());


        //Save the model
        //Note that the saved model will not know which layers were frozen during training.
        //Frozen models always have to specified before training.
        // Models with frozen layers can be constructed in the following two ways:
        //      1. .setFeatureExtractor in the transfer learning API which will always a return a new model (as seen in this example)
        //      2. in place with the TransferLearningHelper constructor which will take a model, and a specific vertexname
        //              and freeze it and the vertices on the path from an input to it (as seen in the FeaturizePreSave class)
        //The saved model can be "fine-tuned" further as in the class "FitFromFeaturized"
//        File locationToSave = new File("MyComputationGraph.zip");
//        boolean saveUpdater = false;
//        ModelSerializer.writeModel(vgg16Transfer, locationToSave, saveUpdater);
//
//        log.info("Model saved");
        log.info("****************Example finished********************");
    }
    private static double evaluateModel(boolean save, String modelPath, ComputationGraph vgg16Transfer, DataSetIterator dataIterVal, double maxACC) throws IOException {


        Evaluation eval = new Evaluation();
        double ACC = 0.0;

        //Evaluation eval = vgg16Transfer.evaluate(dataIterVal);
        eval = vgg16Transfer.evaluate(dataIterVal);

        log.info(eval.stats());

        // Evaluate ACC
        ACC = eval.accuracy();

        log.info("ACC: " + ACC);
        dataIterVal.reset();

        dataIterVal.reset();
        if (ACC > maxACC){
            maxACC = ACC;
            saveModel(save, modelPath, vgg16Transfer);
        }
        return maxACC;
    }


    private static void saveModel(boolean save, String modelPath, ComputationGraph vgg16Transfer) throws IOException{
        if (save) {
            log.info("Save model to: " + modelPath);
            ModelSerializer.writeModel(vgg16Transfer, modelPath , true);
        }
    }
}
