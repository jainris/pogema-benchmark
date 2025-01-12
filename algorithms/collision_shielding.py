import torch

class BaseCollisionShielding:
    def __init__(self, model, env, sampling_method="deterministic"):
        self.model = model
        self.env = env
        self.sampling_method = sampling_method

    def get_actions(self, gdata):
        raise NotImplementedError


class NaiveCollisionShielding(BaseCollisionShielding):
    def __init__(self, model, env, sampling_method="deterministic"):
        super().__init__(model, env, sampling_method)

    def get_actions(self, gdata):
        # Naive collision shielding leaves the shielding to the env
        # So just returning the actions given by the model
        actions = self.model(gdata.x, gdata)
        if self.sampling_method == "deterministic":
            actions = torch.argmax(actions, dim=-1).detach().cpu()
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}.")
        return actions


def get_collision_shielded_model(model, env, args):
    collision_shielding = "naive"
    if "collision_shielding" in vars(args):
        collision_shielding = args.collision_shielding
    if collision_shielding == "naive":
        return NaiveCollisionShielding(
            model=model, env=env, sampling_method=args.action_sampling
        )
    else:
        raise ValueError(
            f"Unsupported collision shielding method: {collision_shielding}."
        )
