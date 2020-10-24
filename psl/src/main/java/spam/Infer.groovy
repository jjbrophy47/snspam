package spam

// Java imports
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.DecimalFormat
// PSL imports
import org.linqs.psl.config.ConfigBundle
import org.linqs.psl.config.ConfigManager
// database
import org.linqs.psl.database.Partition
import org.linqs.psl.database.DataStore
import org.linqs.psl.database.Database
import org.linqs.psl.database.Queries
import org.linqs.psl.database.loading.Inserter
import org.linqs.psl.database.rdbms.RDBMSDataStore
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver.Type
// data loading
import org.linqs.psl.utils.dataloading.InserterUtils
// model
import org.linqs.psl.groovy.PSLModel
import org.linqs.psl.model.rule.Rule
import org.linqs.psl.model.atom.GroundAtom
import org.linqs.psl.model.term.ConstantType
import org.linqs.psl.model.predicate.Predicate
// inference
import org.linqs.psl.application.inference.LazyMPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult
// evaluation
import org.linqs.psl.utils.evaluation.statistics.RankingScore
import org.linqs.psl.utils.evaluation.statistics.SimpleRankingComparator

/**
 * Infer relational model object.
 *
 * Defines all aspects of the model, loads the data, learns weights,
 * and runs inference.
 *
 * @author Jonathan Brophy
 */
public class Infer {
    private static final String W_PT = "write_pt"
    private static final String R_PT = "read_pt"
    private static final String L_PT = "labels_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m
    private PrintWriter fw

    /**
     * Constructor.
     *
     * @param working_dir folder to store temporary datastore in.
     */
    public Infer(String working_dir) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = working_dir + 'psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        this.fw = new PrintWriter(System.out)
    }

    private void out(String message, def newline=1) {
        String msg = newline == 1 ? '\n' + message : message
        // this.fw.print(msg)
        // this.fw.flush()
    }

    private void time(long t1, def suffix='m') {
        long elapsed = System.currentTimeMillis() - t1

        if (suffix == 's') {
            elapsed /= 1000.0
        }
        else if (suffix == 'm') {
            elapsed /= (1000.0 * 60.0)
        }
        else if (suffix == 'h') {
            elapsed /= (1000.0 * 60.0 * 60)
        }
    }

    /**
     * Extract predicates from the untrained model.
     *
     *@param filename name of the text file with the model rules.
     *@returns tuple of predicate names, list of parameters per predicate,
     *  and the names of the closed predicates.
     */
    private Tuple extract_predicates(String filename) {
        def regex = /\s([a-z]+)\(/
        def file = new File(filename)
        def lines = file.readLines()

        def predicates = []
        for (String line: lines) {
            def line_preds = line.findAll(regex).collect{it.replace('(', '')}
            line_preds = line_preds.collect{it.replace(' ', '')}
            predicates += line_preds
        }
        predicates = predicates.toSet().toList()

        def closed = predicates.findAll{!it.contains('spam')}
        def params = predicates.collect{!it.contains('spmy')\
            && (it.contains('has')) ? 2 : 1}
        return new Tuple(predicates, params, closed)
    }

    /**
     * Specify and add predicate definitions to the model.
     */
    private void define_predicates(predicates, params) {
        ConstantType unique_id = ConstantType.UniqueID
        def sgl = [unique_id]
        def dbl = [unique_id, unique_id]

        for (int i = 0; i < predicates.size(); i++) {
            def pred = predicates[i]
            def type = params[i] == 1 ? sgl : dbl
            this.m.add predicate: pred, types: type
        }
    }

    /**
     * Load model rules from a text file.
     *
     *@param filename name of the text file with the model rules.
     */
    private void define_rules(String filename) {
        this.m.addRules(new FileReader(filename))
    }

    /**
     * Load validation and training predicate data.
     *
     *@param working_dir folder to load data from.
     *@param target_name target node identifier.
     *@param closed list of closed predicate names.
     */
    private void load_data(String working_dir, String target_name, def closed) {
        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)
        Partition labels_pt = this.ds.getPartition(L_PT)

        // load test set comments to be labeled.
        // load_file(working_dir + pre + fold, 'spam', labels_pt)
        load_file(working_dir + target_name + '_nolabel', 'spam', write_pt)
        load_file(working_dir + target_name + '_prior', 'prior', read_pt)

        // load relational data.
        for (def pred: closed) {
            def relation = pred
            def hub = pred.replace('has', '')
            def rel_fname = working_dir + relation + '_id' + '_connections'
            def hub_fname = working_dir + hub + '_id'

            load_file(rel_fname, relation, read_pt)
            load_file(hub_fname, 'spmy' + hub, write_pt)
        }
    }

    /**
     * Loads a tab separated predicate data file. Automatically handles
     * truth and non truth files.
     *
     *@param filename name of the file to load.
     *@param predicate_name name of the predicate to load data for.
     *@param partition parition to load the file into.
     */
    private void load_file(filename, predicate_name, partition) {
        String file = filename + '.tsv'
        def predicate = this.m.getPredicate(predicate_name)

        if (new File(file).exists()) {
            Inserter inserter = this.ds.getInserter(predicate, partition)
            InserterUtils.loadDelimitedDataAutomatic(predicate, inserter, file)
        }
    }

    /**
     * Run inference with the trained model on the test set.
     *
     *@param set of closed predicates.
     *@return a FullInferenceResult object.
     */
    private FullInferenceResult run_inference(closed_preds) {
        System.out.println('inference...')
        long start = System.currentTimeMillis()

        Set<Predicate> closed = closed_preds.collect{this.m.getPredicate(it)}

        Partition write_pt = this.ds.getPartition(W_PT)
        Partition read_pt = this.ds.getPartition(R_PT)

        Database inference_db = this.ds.getDatabase(write_pt, closed, read_pt)
        LazyMPEInference mpe = new LazyMPEInference(this.m, inference_db,
                this.cb)
        FullInferenceResult result = mpe.mpeInference()
        mpe.close()
        mpe.finalize()
        inference_db.close()

        time(start)
        System.out.println('done.')
        return result
    }

    /**
     * Write the relational model predictions for each comment in the test set.
     *
     *@param working_dir folder to save predictions to.
     *@param target_id target node identifier.
     */
    private void write_predictions(String working_dir, String target_name) {
        long start = System.currentTimeMillis()

        Partition temp_pt = this.ds.getPartition('temp_pt')
        Partition write_pt = this.ds.getPartition(W_PT)
        Database predictions_db = this.ds.getDatabase(temp_pt, write_pt)

        DecimalFormat formatter = new DecimalFormat("#.#####")
        FileWriter fw = new FileWriter(working_dir + 'psl_scores.tsv')

        fw.write(target_name + '\ty_hat\n')
        for (GroundAtom atom : Queries.getAllAtoms(predictions_db, spam)) {
            double pred = atom.getValue()
            String target_id = atom.getArguments()[0].toString().replace("'", "")
            fw.write(target_id + '\t' + formatter.format(pred) + '\n')
        }
        fw.close()
        predictions_db.close()

        time(start)
    }

    /**
     * Method to define the model, learn weights, and perform inference.
     *
     *@param working_dir temporary working directory.
     *@param target_name target node identifier.
     */
    private void run(String working_dir, String target_name) {
        String rules_filename = working_dir + 'rules_fitted.txt'

        def (predicates, params, closed) = extract_predicates(rules_filename)
        define_predicates(predicates, params)
        define_rules(rules_filename)
        load_data(working_dir, target_name, closed)
        run_inference(closed)
        write_predictions(working_dir, target_name)

        this.ds.close()
    }

    /**
     * Check and parse commandline arguments.
     *
     *@param args arguments from the commandline.
     *@return commandlind args
     */
    public static Tuple check_commandline_args(String[] args) {
        if (args.length < 2) {
            print('Missing args, example: [working_dir] [target_name]')
            System.exit(0)
        }
        String working_dir = args[0].toString()
        String target_name = args[1].toString()
        return new Tuple(working_dir, target_name)
    }

    /**
     * Main method that creates and runs the Infer object.
     *
     *@param args commandline arguments.
     */
    public static void main(String[] args) {
        def (working_dir, target_name) = check_commandline_args(args)
        Infer b = new Infer(working_dir)
        b.run(working_dir, target_name)
    }
}
