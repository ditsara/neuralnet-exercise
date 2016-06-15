module NeuralNet
  class Gate
    class << self
      def forward_multiply(x, y)
        x * y
      end

      def forward_add(x, y)
        x + y
      end

      def forward_sig(x)
        1 / (1 + Math.exp(-x))
      end

    end

    class Multiply
      attr_accessor :u0, :u1, :uout

      def initialize(u0, u1)
        @u0, @u1 = [u0, u1]
        @uout = NeuralNet::Gate::Unit.new(
          NeuralNet::Gate.forward_multiply(u0.val, u1.val),
          0.0)
      end

      def backward
        u0.grad += u1.val * uout.grad
        u1.grad += u0.val * uout.grad
      end
    end

    class Add
      attr_accessor :u0, :u1, :uout

      def initialize(u0, u1)
        @u0, @u1 = [u0, u1]
        @uout = NeuralNet::Gate::Unit.new(
          NeuralNet::Gate.forward_add(u0.val, u1.val),
          0.0)
      end

      def backward
        u0.grad += 1 * uout.grad
        u1.grad += 1 * uout.grad
      end
    end

    class Sigmoid
      attr_accessor :u0, :u1, :uout

      def initialize(u0)
        @u0 = u0
        @uout = NeuralNet::Gate::Unit.new(
          NeuralNet::Gate.forward_sig(u0.val),
          0.0)
      end

      def backward
        s = NeuralNet::Gate.forward_sig(u0.val)
        u0.grad += (s * (1 - s)) * uout.grad
      end
    end

    class Neuron
      # unit inputs
      attr_accessor :a, :b, :c, :x, :y
      # intermediate gates
      attr_accessor :ax, :by, :ax_by, :ax_by_c, :s_ax_by_c
      # unit output
      attr_accessor :uout

      def initialize(a, b, c, x, y)
        @a, @b, @c, @x, @y = [a, b, c, x, y]

        @ax = NeuralNet::Gate::Multiply.new(a, x)
        @by = NeuralNet::Gate::Multiply.new(b, y)

        @ax_by = NeuralNet::Gate::Add.new(ax.uout, by.uout)
        @ax_by_c = NeuralNet::Gate::Add.new(ax_by.uout, c)

        @s_ax_by_c = NeuralNet::Gate::Sigmoid.new(ax_by_c.uout)
        @uout = @s_ax_by_c.uout
      end

      def backward(set_grad: nil)
        @uout.grad = set_grad if set_grad
        
        @s_ax_by_c.backward
        @ax_by_c.backward
        @ax_by.backward
        @by.backward
        @ax.backward
      end
    end

    class Unit
      attr_accessor :val, :grad

      def initialize(val, grad)
        @val, @grad = [val, grad]
      end
    end

  end
end
