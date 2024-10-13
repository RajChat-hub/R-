public class MyJRE {
    public static void main(String[] args) {
        try {
            ProcessBuilder pb = new ProcessBuilder(java, HelloWorld);
            pb.inheritIO();
            pb.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
