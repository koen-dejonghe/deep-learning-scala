package botkop.nn;

import org.nd4j.linalg.api.ndarray.INDArray;

import static org.nd4j.linalg.factory.Nd4j.*;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class TestPerfJava {

    public static void main(String [ ] args) {

        Long t0 = System.currentTimeMillis();
        for (int i=0; i < 25000; i++) {
            sigmoid(randn(764, 30).mmul(randn(30, 10)));
        }
        Long t1 = System.currentTimeMillis();
        System.out.println(t1 - t0);


        Long t2 = System.currentTimeMillis();
        for (int i=0; i < 25000; i++) {
            INDArray x = randn(764, 30);
            INDArray y = randn(30, 10);
            INDArray z = sigmoid(x.mmul(y));
        }
        Long t3 = System.currentTimeMillis();
        System.out.println(t3 - t2);

    }
}
