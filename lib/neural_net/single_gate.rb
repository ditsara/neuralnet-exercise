module NNE
  module SingleGate
    def self.lsearch(x: -2, y: 3, h: 0.0001)
      out = NeuralNet::Gate.forward_multiply(x,y)
      puts "start x: #{x} / y: #{y} / result: #{out} / using h: #{h}"

      best_result = 100.times.inject(
        out: -Float::INFINITY, x: x, y: y
      ) do |best, i|
        x_try = x + (Random.rand * 2 - 1) * h
        y_try = y + (Random.rand * 2 - 1) * h
        out_try = NeuralNet::Gate.forward_multiply x_try, y_try
        out_try > best[:out] ? { out: out_try, x: x_try, y: y_try } : best
      end

      puts "best_x: #{best_result[:x]} / best_y: #{best_result[:y]}"
      best_result[:out]
    end

    def self.numgrad(x: -2, y: 3, h: 0.0001)
      out = NeuralNet::Gate.forward_multiply(x,y)
      puts "start x: #{x} / y: #{y} / result: #{out} / using h: #{h}"

      out_dx = NeuralNet::Gate.forward_multiply(x + h, y)
      dx_h = (out_dx - out) / h
      puts "out_dx: #{out_dx} / df_x: #{dx_h}"

      out_dy = NeuralNet::Gate.forward_multiply(x, y + h)
      dy_h = (out_dy - out) / h
      puts "out_dy: #{out_dy} / df_y: #{dy_h}"

      NeuralNet::Gate.forward_multiply(x + h * dx_h, y + h * dy_h)
    end

    def self.agrad(x: -2, y: 3, h: 0.0001)
      out = NeuralNet::Gate.forward_multiply(x,y)
      puts "start x: #{x} / y: #{y} / result: #{out} / using h: #{h}"

      # derived from the magic of calculus
      x_grad = y
      y_grad = x

      new_x = x + h * x_grad
      new_y = y + h * y_grad

      NeuralNet::Gate.forward_multiply new_x, new_y
    end
  end
end
