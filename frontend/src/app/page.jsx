import Navbar from "../components/navbar";
import DashboardLayout from "../components/dashboard-layout";
// sample data for charts

export default async function App() {
  await new Promise((resolve) => setTimeout(resolve, 4000));
  return (
    <div className="flex h-screen flex-col bg-white">
      <Navbar />
      <DashboardLayout />
    </div>
  );
}
