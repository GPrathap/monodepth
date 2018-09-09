


def wasserstein_gradient_penalty(
    real_data,
    generated_data,
    generator_inputs,
    discriminator_fn,
    discriminator_scope,
    epsilon=1e-10,
    target=1.0,
    one_sided=False,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):

  with ops.name_scope(scope, 'wasserstein_gradient_penalty',
                      (real_data, generated_data)) as scope:
    real_data = ops.convert_to_tensor(real_data)
    generated_data = ops.convert_to_tensor(generated_data)
    if real_data.shape.ndims is None:
      raise ValueError('`real_data` can\'t have unknown rank.')
    if generated_data.shape.ndims is None:
      raise ValueError('`generated_data` can\'t have unknown rank.')

    differences = generated_data - real_data
    batch_size = differences.shape[0].value or array_ops.shape(differences)[0]
    alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
    alpha = random_ops.random_uniform(shape=alpha_shape)
    interpolates = real_data + (alpha * differences)

    with ops.name_scope(None):  # Clear scope so update ops are added properly.
      # Reuse variables if variables already exists.
      with variable_scope.variable_scope(discriminator_scope, 'gpenalty_dscope',
                                         reuse=variable_scope.AUTO_REUSE):
        disc_interpolates = discriminator_fn(interpolates, generator_inputs)

    if isinstance(disc_interpolates, tuple):
      # ACGAN case: disc outputs more than one tensor
      disc_interpolates = disc_interpolates[0]

    gradients = gradients_impl.gradients(disc_interpolates, interpolates)[0]
    gradient_squares = math_ops.reduce_sum(
        math_ops.square(gradients), axis=list(range(1, gradients.shape.ndims)))
    # Propagate shape information, if possible.
    if isinstance(batch_size, int):
      gradient_squares.set_shape([
          batch_size] + gradient_squares.shape.as_list()[1:])
    # For numerical stability, add epsilon to the sum before taking the square
    # root. Note tf.norm does not add epsilon.
    slopes = math_ops.sqrt(gradient_squares + epsilon)
    penalties = slopes / target - 1.0
    if one_sided:
      penalties = math_ops.maximum(0., penalties)
    penalties_squared = math_ops.square(penalties)
    penalty = losses.compute_weighted_loss(
        penalties_squared, weights, scope=scope,
        loss_collection=loss_collection, reduction=reduction)

    if add_summaries:
      summary.scalar('gradient_penalty_loss', penalty)

    return penalty

