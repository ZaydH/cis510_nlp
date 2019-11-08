// Wrapper for maximum-entropy training

// Adapted from Ralph Grishman's code

// invoke by:  java  MEtrain  dataFile  modelFile

// dataFile is the path to your feature-enhanced file
// modelFile is the filename to store the trained model

import java.io.*;
import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;

public class MEtrain {

    public static void main (String[] args) {
	if (args.length != 2) {
	    System.err.println ("MEtrain requires 2 arguments:  dataFile modelFile");
	    System.exit(1);
	}
	String dataFileName = args[0];
	String modelFileName = args[1];
	try {
	    // read events with tab-separated features
	    FileReader datafr = new FileReader(new File(dataFileName));
	    EventStream es = new BasicEventStream(new PlainTextByLineDataStream(datafr), "\t");
	    // train model using 100 iterations, ignoring events occurring fewer than 4 times
	    GISModel model = GIS.trainModel(es, 100, 4);
	    // save model
	    File outputFile = new File(modelFileName);
	    GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
	    writer.persist();
	} catch (Exception e) {
	    System.out.print("Unable to create model due to exception: ");
	    System.out.println(e);
	}
    }
}
