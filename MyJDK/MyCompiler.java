import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

public class MyCompiler {
    public static void main(String[] args) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        int result = compiler.run(null, null, null, HelloWorld.java);
        System.out.println(Compilation Result:  + result);
    }
}
