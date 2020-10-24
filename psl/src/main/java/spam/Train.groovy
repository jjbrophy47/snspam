package spam

// Java imports
import java.io.File
import java.io.FileWriter
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
// weight learning
import org.linqs.psl.application.learning.weight.em.HardEM

/**
 * Train a relational model object.
 *
 * Defines all aspects of the model, loads the data, learns weights,
 * and runs inference.
 *
 * @author Jonathan Brophy
 */
public class Train {
    private static final String WL_W_PT = "wl_write_pt"
    private static final String WL_R_PT = "wl_read_pt"
    private static final String WL_L_PT = "wl_labels_pt"

    private ConfigBundle cb
    private DataStore ds
    private PSLModel m
    private File fw

    /**
     * Constructor.
     *
     * @param working_dir folder to store temporary datastore in.
     */
    public Train(String working_dir) {
        ConfigManager cm = ConfigManager.getManager()

        Date t = new Date()
        String time_of_day = t.getHours() + '_' + t.getMinutes() + '_' +
                t.getSeconds()
        String db_path = working_dir + 'psl_' + time_of_day
        H2DatabaseDriver d = new H2DatabaseDriver(Type.Disk, db_path, true)

        this.cb = cm.getBundle('spam')
        this.ds = new RDBMSDataStore(d, this.cb)
        this.m = new PSLModel(this, this.ds)
        this.fw = new File(working_dir + 'psl_training_log.txt')
        this.fw.append('\ndata store setup at: ' + db_path)
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
        System.out.println(this.m)
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
        System.out.println(predicates)

        def closed = predicates.findAll{it.contains('prior') || it.contains('has')}
        def params = predicates.collect{it.contains('has') ? 2 : 1}
        System.out.println(closed)
        System.out.println(params)

        return new Tuple(predicates, params, closed)
    }

    /**
     * Load validation and training predicate data.
     *
     *@param working_dir folder to load data from.
     *@param target_name target node identifier.
     *@closed list of closed predicate names.
     */
    private void load_data(String working_dir, String target_name, def closed) {
        Partition wl_write_pt = this.ds.getPartition(WL_W_PT)
        Partition wl_read_pt = this.ds.getPartition(WL_R_PT)
        Partition wl_labels_pt = this.ds.getPartition(WL_L_PT)

        // load test set comments to be labeled.
        load_file(working_dir + target_name + '_label', 'spam', wl_labels_pt)
        load_file(working_dir + target_name + '_nolabel', 'spam', wl_write_pt)
        load_file(working_dir + target_name + '_prior', 'prior', wl_read_pt)

        // load relational data.
        for (def pred: closed) {

            if (!pred.contains('has')) {
                continue
            }

            // closed predicates example = [hasburst, prior]

            def hub = pred.replace('has', '')
            def rel_fname = working_dir + hub + '_id_connections'
            def hub_fname = working_dir + hub + '_id'

            load_file(rel_fname, pred, wl_read_pt)
            load_file(hub_fname, 'spmy' + hub, wl_write_pt)
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
        System.out.println(file)
        def predicate = this.m.getPredicate(predicate_name)

        if (new File(file).exists()) {
            Inserter inserter = this.ds.getInserter(predicate, partition)
            InserterUtils.loadDelimitedDataAutomatic(predicate, inserter, file)
        }
    }

    /**
     * Learn weights for model rules using vlidation data.
     *
     *@param closed set of closed predicates.
     */
    private void learn_weights(closed_preds) {
        Set<Predicate> closed = closed_preds.collect{this.m.getPredicate(it)}
        Set<Predicate> closed_labels = [spam]

        Partition wl_wr_pt = ds.getPartition(WL_W_PT)
        Partition wl_r_pt = ds.getPartition(WL_R_PT)
        Partition wl_l_pt = ds.getPartition(WL_L_PT)

        this.fw.append('\nlearning weights...')
        long start = System.currentTimeMillis()

        Database wl_tr_db = this.ds.getDatabase(wl_wr_pt, closed, wl_r_pt)
        Database wl_l_db = ds.getDatabase(wl_l_pt, closed_labels)

        HardEM w_learn = new HardEM(this.m, wl_tr_db, wl_l_db, this.cb)
        w_learn.learn()
        wl_tr_db.close()
        wl_l_db.close()

        long end = System.currentTimeMillis()
        this.fw.append(((end - start) / 60000.0) + 'm')
    }

    /**
     * Write the model with learned weights to a text file.
     *
     *@param working_dir temporary directory to save model to.
     */
    private void write_model(String working_dir) {
        FileWriter mw = new FileWriter(working_dir + 'rules_fitted.txt')
        for (Rule rule : this.m.getRules()) {
            String rule_str = rule.toString().replace('~( ', '~').toLowerCase()
            String rule_filtered = rule_str.replace('( ', '').replace(' )', '')
            this.fw.append('\n\t' + rule_str)
            mw.write(rule_filtered + '\n')
        }
        this.fw.append('\n')
        mw.close()
    }

    /**
     * Method to define the model, learn weights, and perform inference.
     *
     *@param fold experiment identifier.
     *@param data_f data folder.
     *@param pred_f predictions folder.
     *@param model_f model folder.
     */
    private void run(String working_dir, String target_name) {
        String rules_filename = working_dir + 'rules.txt'

        def (predicates, params, closed) = extract_predicates(rules_filename)
        define_predicates(predicates, params)
        define_rules(rules_filename)
        load_data(working_dir, target_name, closed)
        learn_weights(closed)
        write_model(working_dir)

        this.ds.close()
    }

    /**
     * Check and parse commandline arguments.
     *
     *@param args arguments from the commandline.
     *@return a tuple containing the experiment id and social network.
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
     * Main method that creates and runs the Train object.
     *
     *@param args commandline arguments.
     */
    public static void main(String[] args) {
        def (working_dir, target_name) = check_commandline_args(args)
        Train b = new Train(working_dir)
        b.run(working_dir, target_name)
    }
}
