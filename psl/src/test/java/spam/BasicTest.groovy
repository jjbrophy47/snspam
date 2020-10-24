package spam

// Java imports
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
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
import org.linqs.psl.cli.dataloader.DataInserter
// model
import org.linqs.psl.groovy.PSLModel
import org.linqs.psl.model.rule.Rule
import org.linqs.psl.model.atom.GroundAtom
import org.linqs.psl.model.term.ConstantType
import org.linqs.psl.model.predicate.Predicate
// weight learning
import org.linqs.psl.application.learning.weight.em.HardEM
// inference
import org.linqs.psl.application.inference.MPEInference
import org.linqs.psl.application.inference.result.FullInferenceResult

public class BasicTest {
    private static final String W_PT = "write_pt"
    private static final String R_PT = "read_pt"
    private static final String L_PT = "labels_pt"
    private static final String WL_W_PT = "wl_write_pt"
    private static final String WL_R_PT = "wl_read_pt"
    private static final String WL_L_PT = "wl_labels_pt"

    private Basic test_obj

    @Before
    public void setup() {
        test_obj = new Basic('data/')
    }

    @Test
    public void testDefineFileFolders() {
    }
}
