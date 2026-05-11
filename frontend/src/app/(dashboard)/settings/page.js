"use client";

import { UserProfile } from "@clerk/nextjs";

const appearance = {
  variables: {
    colorPrimary: "#7c3aed",
    borderRadius: "0.75rem",
  },
}

export default function SettingsPage() {
  return (
    <div className="p-6 md:p-8 max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Manage your profile, connected accounts, and security.
        </p>
      </div>
      <UserProfile appearance={appearance} routing="hash" />
    </div>
  );
}
