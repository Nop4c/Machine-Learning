import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


public class Main {

	public static void main(String[] args) throws IOException {
		int height = 28;
		int width = 28;
		int channels = 1;
		int rngseed = 123;
		Random randNumGen = new Random(rngseed);
		int batchSize = 1;
		int outputNum = 10;

		File trainData = new File("C:\\Users\\Theo\\git\\Machine-Learning\\MachineLearning\\train");
		File testData = new File("C:\\Users\\Theo\\git\\Machine-Learning\\MachineLearning\\test");
		
		
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
		
		recordReader.initialize(train);
		recordReader.setListeners(new LogRecordListener());
		
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		
		for(int i = 1; i< 3; i++) {
			DataSet ds = dataIter.next();
			System.out.println(ds);
			System.out.println(dataIter.getLabels());
		}
	}

}
