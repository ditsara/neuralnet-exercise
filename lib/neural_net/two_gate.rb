module NNE
  module TwoGate
    def self.forward_circuit(x: -2, y: 5, z: -4)
      q = NeuralNet::Gate.forward_add x, y
      NeuralNet::Gate.forward_multiply q, z
    end

    def self.chain_rule(x: -2, y: 5, z: -4, h: 0.0001)
      q = NeuralNet::Gate.forward_add x, y
      f = NeuralNet::Gate.forward_multiply q, z

      df_z = q
      df_q = z

      dq_x = 1
      dq_y = 1

      df_x = dq_x * df_q
      df_y = dq_y * df_q

      gradf_xyz = [df_x, df_y, df_z]
      puts "gradf_xyz: #{gradf_xyz}"

      new_x = x + h * df_x
      new_y = y + h * df_y
      new_z = z + h * df_z

      forward_circuit x: new_x, y: new_y, z: new_z
    end
  end
end
