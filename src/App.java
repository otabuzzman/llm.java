import com.otabuzzman.llmj.DataLoader;
import com.otabuzzman.llmj.GPT2;
import com.otabuzzman.llmj.Glob;
import com.otabuzzman.llmj.Mt19937;
import com.otabuzzman.llmj.Tokenizer;

public class App {
    public static void main(String[] args) throws Exception {
        Tokenizer tokenizer = new Tokenizer("gpt2_tokenizer.bin");
        Glob glob = new Glob("**/*.java", ".");
        for ( int i=0 ; i< glob.gl_pathc() ; i++ ) {
            System.out.println(glob.gl_pathv(i));
        }
        Mt19937.test_Mt19937();
        DataLoader loader = new DataLoader("tiny_shakespeare_train.bin", 4, 64, 0, 1, true);
        GPT2 gpt2 = new GPT2("gpt2_124M.bin");
        System.out.println("Hello, World!");
    }
}
