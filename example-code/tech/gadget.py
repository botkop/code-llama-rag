class Gadget:
    def __init__(self, model, performance_level):
        self.model = model
        self.performance_level = performance_level

    def overclock(self):
        """Magically enhances the gadget's performance level."""
        self.performance_level += 1
        print(f"{self.model} is now overclocked! Performance level: {self.performance_level}")
    
    def check_status(self):
        """Checks the current status of the gadget."""
        return f"Model: {self.model}, Performance Level: {self.performance_level}"

# Using the Gadget
my_gadget = Gadget("Techie Gizmo 3000", 5)
print("Original status:", my_gadget.check_status())

# Overclocking the gadget to enhance its performance
my_gadget.overclock()
print("After overclocking:", my_gadget.check_status())

