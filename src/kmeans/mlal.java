package kmeans;
import java.io.*;
import java.util.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.*;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.*;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;
public class mlal{  
	public static void main(String[] args) throws Exception  
	{   
		LinkedList<String> l = new LinkedList<String>();//List to store the elements of y train and y test
		LinkedList<String> lx = new LinkedList<>();// list to store the elements of x_train and x_test
		Scanner sc = new Scanner(new File("C:\\Users\\Akash\\eclipse-workspace\\kmeans\\datasets\\fasttoslow.csv"));//reading the file (x_train and x_test)
		Scanner sc1 = new Scanner(new File("C:\\Users\\Akash\\eclipse-workspace\\kmeans\\datasets\\x_trainfull.csv"));// provide the location of the file 
		String s =new String();
		ArrayList<Integer> new_index=new ArrayList<Integer>();// index
		List<Integer> y_train = new ArrayList<Integer>();// list to store y_train
		List<Integer> y_test = new ArrayList<Integer>();//list to store y_test
		List<List<String>> x_train = new ArrayList<>();//list of list to store x_train
		List<List<String>> x_test = new ArrayList<>();//list of list to store x_test
		List<Integer> index_train = new ArrayList<>();
		List<Integer> index_test = new ArrayList<>();
		ArrayList<Integer> new_y_train=new ArrayList<Integer>();
		ArrayList<Integer> small_cat=new ArrayList<Integer>(Arrays.asList(2,3,6,7,10));
		ArrayList<Integer> frequency=new ArrayList<Integer>(Arrays.asList(0,0,0,0,0,0,0,0,0,0,0,0,0,0));
		ArrayList<Integer> frequency3=new ArrayList<Integer>(Arrays.asList(0,0,0,0,0,0,0,0,0,0,0,0,0,0));
		int counter=0;
		while (sc.hasNext())  
		{  
			l.add(sc.next()); 


		} 
		while(sc1.hasNext()) {
			lx.add(sc1.next());
		}
		int c=600;// spliting the dataset
		int d=808;// total number of rows n a dataset
		for (int i=1; i<c ;i++) {
			System.out.println(i);
			//			System.out.println(lx.get(i));
			s=lx.get(i);
			String[] strSplit=s.split(" ");
			ArrayList<String> a=new ArrayList<String>(Arrays.asList(strSplit));
			//			System.out.println(a);
			x_train.add(a);
			System.out.println("x_train values"+x_train);
		}
		for (int i=c; i<d ;i++) {
			System.out.println(i);
			//			System.out.println(lx.get(i));
			s=lx.get(i);
			String[] strSplit=s.split(" ");
			ArrayList<String> a=new ArrayList<String>(Arrays.asList(strSplit));
			//			System.out.println(a);
			x_test.add(a);
			System.out.println("X_test values are"+x_test);
		}
		for (int i=1; i<l.size() ;i++) {
			//			System.out.println(i);
			//			System.out.println(l.get(i));
			s=l.get(i);
			String[] strSplit=s.split(",");
			ArrayList<String> a=new ArrayList<String>(Arrays.asList(strSplit));
			//			System.out.println(a);
			//			System.out.println(a.get(0));
			int temp= Integer.parseInt(a.get(0));//checking the fastest identifier in the data set.
			if (temp>=13) {
				new_index.add(temp);
			}
			else {
				new_index.add(16);
			}
			for(int j=0;j<a.size();j++) {
				int val=Integer.parseInt(a.get(j));
				if(counter<1) {
					if(val<14) {
						frequency.set(val,frequency.get(val)+1);
					}
				}
				if(counter<3) {
					if(val<14) {
						frequency3.set(val,frequency3.get(val)+1);
					}
				}
				counter+=1;
				if(val<13) {
					for(int k=0;k<small_cat.size();k++) {
						if(small_cat.get(k)==val) {
							new_y_train.add(val);
							break;
						}
					}
					break;
				}
			}


		}
		//		System.out.println(new_y_train);
		int sample_y_train[][] = new int[d][1]; //transpose of new_y_train matrix
		for(int i=0;i<new_y_train.size();i++) {
			sample_y_train[i][0]=new_y_train.get(i);
		}
		int sample_index_train[][] = new int[d][1]; //transpose of new_y_train matrix
		for(int i=0;i<new_index.size();i++) {
			sample_index_train[i][0]=new_index.get(i);
		} 

		//		System.out.println(new_index);
		y_train = new_y_train.subList(0, c);
		y_test = new_y_train.subList(c,d);
		//		x_train = lfinal.subList(0, 600);
		//		x_test = lx.subList(600,808);
		//		System.out.println(x_test);
		index_train = new_index.subList(0, c);// spliting the obtained indexes
		index_test = new_index.subList(c,d);

		callBDT(x_test,index_test,l);// calling the BDT
		//process();
		decisiontree();
		sc.close();
	} 

	public static ArrayList<Integer> BDT(List<List<String>> features) {
		ArrayList<Integer> predict = new ArrayList<>();
		List<String> fea;
		for(int i=0;i<features.size();i++) {
			fea=features.get(i);
			String s = fea.get(0);
			String[] strSplit=s.split(",");
			ArrayList<String> a=new ArrayList<String>(Arrays.asList(strSplit));
			int val1=Integer.parseInt(a.get(1));// accesing the dimension.
			int val0=Integer.parseInt(a.get(0));// accessing the k.
			if(val1<20) {
				predict.add(13);//predicts fastest id as 13 (index)
			}
			else {
				if(val0>50) {
					predict.add(6);//predicts fastest identifier as 6 (Hame)
				}else {
					predict.add(2);//predicts fastest identifier as 2 (Yinyang)
				}
			}
		}
		return predict;
	}
	public static void callBDT(List<List<String>> x_test, List<Integer> index_test, LinkedList<String> l) {
		ArrayList<Integer> predy = BDT(x_test);
		System.out.println("the values predicted by BDT is"+predy);
		float dtboundP=mrr_bound(predy,index_test,l);
		System.out.println("\nthe mean reciprocal rank value of BDT is"+dtboundP);

	}
	public static float mrr_bound(ArrayList<Integer> predy, List<Integer> index_test, LinkedList<String> l) {
		float sum=(float) 0.0;
		int ci=0;
		System.out.print("groundtruth values are"+index_test);
		for(int i=1;i<index_test.size();i++) {
			int val=index_test.get(i);
			String s = l.get(val);
			int predict_y = predy.get(ci);
			int counts=1;
			ci+=1;
			String[] strSplit=s.split(",");
			ArrayList<String> a=new ArrayList<String>(Arrays.asList(strSplit));
			for(int j=0;j<a.size();j++) {
				int eachy=Integer.parseInt(a.get(j));
				if(predict_y==eachy) {
					sum=(sum+1)/counts;
					break;
				}
				if(eachy<13) {
					counts=counts+1;
				}
			}

		}
		return sum;
	}
	public static void decisiontree() throws Exception
{
		int classIdx = 1;
		CSVLoader loader=new CSVLoader();// loading csv file
	    //Instances train =(new DataSource(file)) .getDataSet();
		loader.setFile(new File("C:\\Users\\Akash\\eclipse-workspace\\kmeans\\datasets\\x_train1.csv"));
	    Instances train1 = loader.getDataSet();// Getting data set
		train1.setClassIndex(classIdx);
		NumericToNominal convert =new NumericToNominal(); // j48 can work on nominal values 
		String[] options= new String[2];
		options[0]="-R";
		options[1]="1-9";  //range of variables to make numeric
		convert.setOptions(options);
		convert.setInputFormat(train1);
		Instances train=Filter.useFilter(train1, convert);
		System.out.println(train.numInstances());
		//train.randomize(new java.util.Random());
		int trainSize = (int) Math.round(train.numInstances() * 0.8);// spliting the dataset in 80-20 ratio
		int testSize = train.numInstances() - trainSize;
		System.out.println(trainSize);
		Instances x_train = new Instances(train, 0, trainSize);
		Instances x_test = new Instances(train, trainSize, testSize);
		//Instances test= new Instances();
		System.out.println(x_test);
		Classifier classifier = new J48();
		classifier.buildClassifier(x_train);// building the classifier
		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(x_train);
		eval.evaluateModel(classifier, x_test);// evaluating
		//predict=classifier.classifyInstance(x_test);
		System.out.println(classifier);
		System.out.println(eval.toMatrixString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		
	}



}

