module NNE

  def self.sig(a: 1.0, b: 2.0, c: -3.0, x: -1.0, y: 3.0, h: 0.01)
    puts "inputs: #{[a,b,c,x,y]}"
    ua, ub, uc, ux, uy = [a, b, c, x, y]
      .map { |val| NeuralNet::Gate::Unit.new(val, 0.0) }

    out_0 = NeuralNet::Gate::Neuron.new(ua, ub, uc, ux, uy)
    puts "initial value: #{out_0.uout.val}"

    puts "step size: #{h}"
    out_0.backward set_grad: 1.0
    ua.val = a + h * out_0.a.grad
    ub.val = b + h * out_0.b.grad
    uc.val = c + h * out_0.c.grad
    ux.val = x + h * out_0.x.grad
    uy.val = y + h * out_0.y.grad

    out_1 = NeuralNet::Gate::Neuron.new(ua, ub, uc, ux, uy)
    puts "value after one backprop: #{out_1.uout.val}"

    out_1
  end
end
